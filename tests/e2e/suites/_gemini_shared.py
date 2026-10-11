"""
Group ``shared`` of the gemini suite: image editing in chats that Open WebUI
0.12 shares with "Allow replies" (google_gemini.py 1.20.0). Run by
suites/gemini.py; the leading underscore keeps this module out of the suite
discovery.

Open WebUI 0.12 shares a chat with users or groups (``POST
/api/v1/chats/<id>/share`` with ``share_mode: "continue"``, then ``POST
/api/v1/chats/shared/<id>/access/update`` with the grants) or a folder (``POST
/api/v1/folders/<id>/access/update``). The participants of a "continue" share
write in the owner's chat: chat_completion (main.py) asks
``Chats.get_accessible_chat_by_id(chat_id, user, permission="write")`` before
it calls the pipe. A share without "continue" (Clone only) and a read-only
folder share give read access only. The pipe loads the image history of a chat
with the same check and reads a file of another user only for an admin or when
``has_access_to_file(file_id, "read", user)`` lets the user read it (Open
WebUI's own rule, routers/files.py). It links its generated images to the chat
(chat_file) like Open WebUI's own image generation, so the participants may read
them; Open WebUI links uploads itself.

Users (all non-admin): owner A, participant B (user grant on chat X), group
member G (group grant on chat X), viewer D (chat Y shared Clone only with D,
and Y in a folder shared read-only with D) and outsider C. Chat X: A's turn
with the upload "upload1" (generated image "final"), then B, A and G write in
it. Chat Y: two turns of A (upload "upload2", generated "gen1").

Chat X2 of A, shared with "Allow replies" with B, covers the access fixes of
the security review (see _security): a file B names in a message is read on
A's turn only if B may read it too (shared.deputy), the images of earlier
messages looked at per request are capped (shared.history-cap) and generated
images are linked only to a chat the user may continue (shared.link-chat).

The test-only global filter REDIRECT_FILTER hands the pipe another chat in
``__metadata__`` (``e2e-chat=<chat id>/<user message id>`` in the prompt of an
API request; ``e2e-link=<chat id>`` with a new message id, so the pipe emits
and links its images as for a chat message), as if Open WebUI had called it
for that chat without its own check: so the pipe's own check is tested also
for the users Open WebUI turns away before the pipe (the admin, the read-only
viewer, the outsider). Its fault markers make the pipe's next call of an
access helper raise, or (``e2e-chat-truthy=``) the chat check return True
instead of the chat.

On Open WebUI before 0.12 (no "Allow replies") the group records
``shared.unsupported`` (PASS) and ``shared.legacy``: the pipe keeps its owner /
admin rule there (also checked by the imgedit group), although Open WebUI 0.11
lets a user a chat is shared with read its files.
"""

from harness import BrowserSession, Suite, short
from harness.known import version_tuple

from .gemini import (
    FOREIGN_PNG,
    IMAGE_GA,
    LEGACY,
    SAVED_PNG,
    UPLOAD_PNG_1,
    UPLOAD_PNG_2,
    VEO,
    _edit_report,
    _edit_request,
    _gen,
    _grant_read,
    _upload_png,
    _video_image,
)

OWNER = "e2e-shared-owner@example.com"
PARTICIPANT = "e2e-shared-participant@example.com"
MEMBER = "e2e-shared-member@example.com"
VIEWER = "e2e-shared-viewer@example.com"
OUTSIDER = "e2e-shared-outsider@example.com"
GROUP_NAME = "E2E Shared Chats"
FOLDER_NAME = "E2E Shared Folder"
EDIT = "Change the name to Bob"
# what the edit request sends with the history of chat X's first turn
WITH_HISTORY = [EDIT, "[Image 1]", "upload1", "[Image 2]", "final"]
FILE_REFUSED = "does not belong to the requesting user"
# the pipe's refusals of a file (any reason) and of one named by another user
ANY_REFUSED = "Not reading a file for user"
AUTHOR_REFUSED = "who may not read it"
LINK_REFUSED = "Not linking generated files to chat"
FILE_URL = "/api/v1/files/{}/content"
# images of earlier messages the pipe looks at per request: 4 x
# IMAGE_HISTORY_MAX_REFERENCES (5); shared.history-cap plants more links
HISTORY_CAP = 4 * 5
PLANTED = 30

REDIRECT_FILTER_ID = "e2e_chat_redirect"
REDIRECT_FILTER = '''"""
title: E2E Chat Redirect
author: owndev
version: 0.1.0
license: Apache License 2.0
description: Test-only filter for tests/e2e (gemini shared). In an API request (no message id), "e2e-chat=<chat id>/<user message id>" in the last user message puts that chat id and user message id into the pipe's __metadata__, and "e2e-link=<chat id>" that chat id and a new message id (the pipe then emits and links its images as for a chat message). "e2e-chat-fault=<chat id>" / "e2e-file-fault=<file id>" make the next call of Chats.get_accessible_chat_by_id / has_access_to_file for that id from the Gemini function raise RuntimeError, "e2e-chat-truthy=<chat id>" the next call of Chats.get_accessible_chat_by_id return True (a truthy value that is not the chat), once (within 60 s). The markers are removed from the message.
"""

import functools
import re
import sys
import time
import uuid
from typing import Optional

REDIRECT = re.compile(r"\\s*e2e-chat=([A-Za-z0-9-]+)/([A-Za-z0-9-]+)")
LINK = re.compile(r"\\s*e2e-link=([A-Za-z0-9-]+)")
FAULT = re.compile(r"\\s*e2e-(chat|file)-fault=([A-Za-z0-9-]+)")
TRUTHY = re.compile(r"\\s*e2e-chat-truthy=([A-Za-z0-9-]+)")


def _arm(owner, name: str, key: str, outcome) -> None:
    """Wrap owner.<name> once (an async wrapper with the same signature, so the
    pipe's feature detection still sees the helper; it stays in place, inert
    unless armed) and arm it for ``key`` (its first argument): its next call
    from the Gemini function raises ``outcome`` (an exception) or returns it."""
    wrapper = getattr(owner, name)
    if not hasattr(wrapper, "e2e_outcomes"):
        original = wrapper

        @functools.wraps(original)
        async def wrapper(*args, **kwargs):
            caller = sys._getframe(1).f_globals.get("__name__", "")
            armed = wrapper.e2e_outcomes
            if caller.startswith("function_gemini") and args and args[0] in armed:
                deadline, result = armed.pop(args[0])
                if time.time() < deadline:
                    if isinstance(result, BaseException):
                        raise result
                    return result
            return await original(*args, **kwargs)

        wrapper.e2e_outcomes = {}
        setattr(owner, name, wrapper)
    wrapper.e2e_outcomes[key] = (time.time() + 60, outcome)


class Filter:
    async def inlet(self, body: dict, __metadata__: Optional[dict] = None) -> dict:
        messages = body.get("messages") or []
        text = messages[-1].get("content") if messages else None
        if __metadata__ is None or not isinstance(text, str):
            return body
        from open_webui.models.chats import Chats

        for kind, key in FAULT.findall(text):
            if kind == "chat":
                error = RuntimeError("e2e chat access fault")
                _arm(Chats, "get_accessible_chat_by_id", key, error)
            else:
                from open_webui.utils.access_control import files

                error = RuntimeError("e2e file access fault")
                _arm(files, "has_access_to_file", key, error)
        for key in TRUTHY.findall(text):
            _arm(Chats, "get_accessible_chat_by_id", key, True)
        text = TRUTHY.sub("", FAULT.sub("", text))
        match = REDIRECT.search(text)
        if match and not __metadata__.get("message_id"):
            __metadata__["chat_id"], __metadata__["user_message_id"] = match.groups()
            text = REDIRECT.sub("", text)
        link = LINK.search(text)
        if link and not __metadata__.get("message_id"):
            __metadata__["chat_id"] = link.group(1)
            __metadata__["message_id"] = str(uuid.uuid4())
            text = LINK.sub("", text)
        messages[-1]["content"] = text.strip()
        return body
'''


def _grant(principal_type: str, principal_id: str) -> dict:
    return {
        "principal_type": principal_type,
        "principal_id": principal_id,
        "permission": "read",
    }


async def _share(owner, chat_id: str, grants: list, replies: bool) -> list:
    """Share a chat like the web UI's share dialog: the share link (with
    share_mode "continue" for "Allow replies"), then who may open it.
    Returns the HTTP statuses."""
    mode = {"share_mode": "continue"} if replies else {}
    s1, _ = await owner.api("POST", f"/api/v1/chats/{chat_id}/share", mode)
    s2, _ = await owner.api(
        "POST",
        f"/api/v1/chats/shared/{chat_id}/access/update",
        {"access_grants": grants, **mode},
    )
    return [s1, s2]


async def _group(t: Suite, member_id: str) -> tuple:
    """(group id, HTTP status of adding the member); the group is reused."""
    _, groups = await t.owui.api("GET", "/api/v1/groups/")
    found = [
        g for g in groups or [] if isinstance(g, dict) and g.get("name") == GROUP_NAME
    ]
    if found:
        group_id = found[0]["id"]
    else:
        _, created = await t.owui.api(
            "POST",
            "/api/v1/groups/create",
            {"name": GROUP_NAME, "description": "e2e shared chats"},
        )
        group_id = created.get("id", "") if isinstance(created, dict) else ""
    status, _ = await t.owui.api(
        "POST", f"/api/v1/groups/id/{group_id}/users/add", {"user_ids": [member_id]}
    )
    return group_id, status


async def _read_only_folder(owner, chat_id: str, viewer_id: str) -> list:
    """Move the chat into a folder of the owner shared read-only (no "continue")
    with the viewer. Returns the HTTP statuses."""
    _, folders = await owner.api("GET", "/api/v1/folders/")
    found = [
        f for f in folders or [] if isinstance(f, dict) and f.get("name") == FOLDER_NAME
    ]
    if found:
        s1, folder_id = 200, found[0]["id"]
    else:
        s1, created = await owner.api("POST", "/api/v1/folders/", {"name": FOLDER_NAME})
        folder_id = created.get("id", "") if isinstance(created, dict) else ""
    s2, _ = await owner.api(
        "POST", f"/api/v1/chats/{chat_id}/folder", {"folder_id": folder_id}
    )
    s3, _ = await owner.api(
        "POST",
        f"/api/v1/folders/{folder_id}/access/update",
        {"access_grants": [_grant("user", viewer_id)]},
    )
    return [s1, s2, s3]


async def _file_status(client, file_id: str) -> int:
    return (await client.request("GET", f"/api/v1/files/{file_id}/content")).status_code


def _target(chat) -> str:
    """``e2e-chat=<chat id>/<user message id>`` of a browser turn."""
    return f"e2e-chat={chat.chat_id}/{(chat.message or {}).get('parentId')}"


async def _api_edit(client, mock, text: str) -> tuple:
    """An API request to the image model: (result, what it sent upstream)."""
    await mock.reset()
    result = await client.chat(IMAGE_GA, text, stream=False)
    return result, await _edit_request(mock)


def _brief(chat) -> str:
    if chat is None:
        return "-"
    return f"HTTP {chat.http_status} done={chat.done}"


async def _file_ids(client) -> set:
    """Ids of the user's newest files (the first page of GET /api/v1/files/)."""
    _, data = await client.api("GET", "/api/v1/files/?content=false")
    items = data.get("items") if isinstance(data, dict) else data
    return {f.get("id") for f in items or [] if isinstance(f, dict)}


async def _security(t: Suite, mock, owner, part, outsider, private, admin_file):
    """Chat X2 of the owner A, shared with "Allow replies" with the participant
    B, for the checks of the access fixes in 1.20.0:

    - deputy: B names A's private file in a message of X2. A's next turn must
      not send it (B may not read it, so A's rights must not be used for it);
      A's own link to it, in A's own message, is sent;
    - history cap: B plants PLANTED links to files A may not use (the admin's
      file and A's private file); A's next turn checks at most HISTORY_CAP
      images of earlier messages;
    - linking: the test filter hands the pipe X2 with a new message id; B's
      generated image is linked to X2 (A may read it), the outsider C's
      images are not (C may not continue X2), with and without the decision of
      the image history (``e2e-chat=``).

    Returns what the checks need; ``error`` is set when the flow broke."""
    r: dict = {"error": "", "a3": {}, "a4": {}, "a5": {}, "a7": {}}
    try:
        upload = await _upload_png(owner, UPLOAD_PNG_1, "upload1.png")
        url = FILE_URL.format(private["id"])
        async with BrowserSession(owner) as b:
            await mock.reset()
            a1 = await b.chat(
                IMAGE_GA,
                "Draw a certificate like this",
                params=LEGACY,
                user_files=[upload],
            )
        r["shares"] = await _share(
            owner, a1.chat_id, [_grant("user", part.user["id"])], replies=True
        )
        r["before"] = await _file_status(part, private["id"])
        async with BrowserSession(part) as b:
            await mock.reset()
            b2 = await b.chat(
                IMAGE_GA,
                f"Nice ![p]({url}) final-image-2",
                params=LEGACY,
                chat_id=a1.chat_id,
                parent_id=a1.message_id,
            )
        r["b2"] = _brief(b2)
        await t.log.settle(0.5)
        mark = t.mark()
        async with BrowserSession(owner) as b:
            await mock.reset()
            a3 = await b.chat(
                IMAGE_GA,
                "Make it blue final-image-3",
                params=LEGACY,
                chat_id=a1.chat_id,
                parent_id=b2.message_id,
            )
            r["a3"], r["a3_done"] = await _edit_request(mock), a3.done
            await t.log.settle(0.5)
            r["a3_refused"] = len(t.log.lines(mark, AUTHOR_REFUSED))
            await mock.reset()
            a4 = await b.chat(
                IMAGE_GA,
                f"Use this one ![p]({url}) final-image-1",
                params=LEGACY,
                chat_id=a1.chat_id,
                parent_id=a3.message_id,
            )
            r["a4"] = await _edit_request(mock)
            await mock.reset()
            a5 = await b.chat(
                IMAGE_GA,
                "Make it red",
                params=LEGACY,
                chat_id=a1.chat_id,
                parent_id=a4.message_id,
            )
            r["a5"], r["a5_done"] = await _edit_request(mock), a5.done
        r["after"] = await _file_status(part, private["id"])

        planted = " ".join(
            f"![x]({FILE_URL.format(fid)})"
            for fid in [admin_file["id"], private["id"]] * (PLANTED // 2)
        )
        async with BrowserSession(part) as b:
            await mock.reset()
            b6 = await b.chat(
                IMAGE_GA,
                f"Look {planted} final-image-2",
                params=LEGACY,
                chat_id=a1.chat_id,
                parent_id=a5.message_id,
            )
        r["b6"] = _brief(b6)
        await t.log.settle(0.5)
        mark = t.mark()
        async with BrowserSession(owner) as b:
            await mock.reset()
            a7 = await b.chat(
                IMAGE_GA,
                "Make it green",
                params=LEGACY,
                chat_id=a1.chat_id,
                parent_id=b6.message_id,
            )
            r["a7"], r["a7_brief"] = await _edit_request(mock), _brief(a7)
        await t.log.settle(0.5)
        r["a7_done"] = a7.done
        r["cap_refused"] = len(t.log.lines(mark, ANY_REFUSED))

        before_b, before_c = await _file_ids(part), await _file_ids(outsider)
        mark = t.mark()
        await mock.reset()
        link = f"e2e-link={a1.chat_id}"
        r["link_status"] = [
            getattr(await client.chat(IMAGE_GA, text, stream=False), "status", None)
            for client, text in (
                (part, f"Draw a star {link}"),
                (outsider, f"Draw a star {link}"),
                (outsider, f"Draw a star {_target(a5)} {link}"),
            )
        ]
        await t.log.settle(0.5)
        r["link_refused"] = len(t.log.lines(mark, LINK_REFUSED))
        t.expect_warnings(mark, ("function_gemini", LINK_REFUSED))
        r["owner_reads_b"] = [
            await _file_status(owner, fid)
            for fid in sorted((await _file_ids(part)) - before_b)
        ]
        r["owner_reads_c"] = [
            await _file_status(owner, fid)
            for fid in sorted((await _file_ids(outsider)) - before_c)
        ]
    except Exception as exc:  # noqa: BLE001 - reported in the checks
        r["error"] = f"setup failed: {exc!r} "
    return r


async def legacy(t: Suite, mock, version: str) -> None:
    """Open WebUI before 0.12: no "Allow replies", and has_access_to_file
    (0.11) knows only share links with grants. The pipe keeps its owner / admin
    rule there: a user the chat is shared with reads the chat's files in Open
    WebUI (the generated image too, linked by the pipe), but a markdown link of
    that user to the owner's upload is not read."""
    t.check(
        "shared.unsupported",
        f"shared chats with replies: feature not in Open WebUI {version}",
        True,
        f"owui={version}",
    )
    error, users, grants, shares, visible = "", {}, [], [], {}
    c1 = result = None
    req = {"contents": 0, "sent": [], "images": []}
    denied = 0
    try:
        for key, name, email in (
            ("owner", "E2E Shared Owner", OWNER),
            ("participant", "E2E Shared Participant", PARTICIPANT),
        ):
            users[key] = await t.owui.create_user(name, email)
        owner, part = users["owner"], users["participant"]
        grants = [await _grant_read(t, IMAGE_GA, "Gemini 3.1 Flash Image")]
        await t.owui.models(refresh=True)
        upload = await _upload_png(owner, UPLOAD_PNG_1, "upload1.png")
        async with BrowserSession(owner) as b:
            await mock.reset()
            c1 = await b.chat(
                IMAGE_GA,
                "Draw a certificate like this",
                params=LEGACY,
                user_files=[upload],
            )
        shares = await _share(
            owner, c1.chat_id, [_grant("user", part.user["id"])], replies=False
        )
        generated = [
            (f.get("url") or "").split("/files/")[-1].split("/")[0]
            for f in c1.files
            if f.get("type") == "image"
        ]
        visible = {
            "upload": await _file_status(part, upload["id"]),
            "generated": [await _file_status(part, fid) for fid in generated],
        }
        mark = t.mark()
        result, req = await _api_edit(
            part, mock, f"Edit this ![a](/api/v1/files/{upload['id']}/content)"
        )
        await t.log.settle(0.5)
        denied = len(t.log.lines(mark, FILE_REFUSED))
    except Exception as exc:  # noqa: BLE001 - reported in the check
        error = f"setup failed: {exc!r} "
    finally:
        for client in users.values():
            await client.close()
        await t.owui.delete_model(IMAGE_GA)
    t.check(
        "shared.legacy",
        f"Open WebUI {version}: a user the chat is shared with reads the owner's "
        "upload and generated image in Open WebUI, but the pipe keeps its owner / "
        "admin rule: the user's markdown link to the upload is not read",
        not error
        and grants == [200]
        and shares == [200, 200]
        and visible.get("upload") == 200
        and visible.get("generated") == [200]
        and getattr(result, "status", None) == 200
        and req["images"] == []
        and denied > 0,
        f"{error}grants={grants} shares={shares} turn1: {_brief(c1)} "
        f"participant: upload HTTP {visible.get('upload')} generated HTTP "
        f"{visible.get('generated')} link: HTTP {getattr(result, 'status', None)} "
        f"{_edit_report(req)} denied_logged={denied}",
    )


async def shared(t: Suite, mock) -> None:
    version = await t.owui.version()
    if version_tuple(version) < (0, 12):
        await legacy(t, mock, version)
        return
    empty = {"contents": 0, "sent": [], "images": []}
    error = ""
    users: dict = {}
    grants: list = []
    shares: list = []
    group_status = folder = None
    visible: dict = {}
    cx1 = cb = ca = cg = cy1 = cy2 = cd = cc = None
    rb = ra = rg = empty
    viewer_calls = outsider_calls = -1
    redirected: dict = {}
    links = video = chat_fault = file_fault = None
    links_req = file_fault_req = chat_fault_req = empty
    video_image = ""
    denied = chat_fault_logged = file_fault_logged = 0
    ids_logged: list = []
    group_id = ""
    sec: dict = {"error": "not run ", "a3": {}, "a4": {}, "a5": {}, "a7": {}}
    try:
        for key, name, email in (
            ("owner", "E2E Shared Owner", OWNER),
            ("participant", "E2E Shared Participant", PARTICIPANT),
            ("member", "E2E Shared Member", MEMBER),
            ("viewer", "E2E Shared Viewer", VIEWER),
            ("outsider", "E2E Shared Outsider", OUTSIDER),
        ):
            users[key] = await t.owui.create_user(name, email)
        owner, part = users["owner"], users["participant"]
        grants = [
            await _grant_read(t, IMAGE_GA, "Gemini 3.1 Flash Image"),
            await _grant_read(t, VEO, "Veo 3.1 Generate Preview"),
        ]
        await t.owui.models(refresh=True)
        status, _ = await t.owui.install_function(
            REDIRECT_FILTER_ID, "E2E Chat Redirect", REDIRECT_FILTER
        )
        grants += [
            status,
            await t.owui.set_active(REDIRECT_FILTER_ID, True),
            await t.owui.set_global(REDIRECT_FILTER_ID, True),
        ]
        group_id, group_status = await _group(t, users["member"].user["id"])
        upload = await _upload_png(owner, UPLOAD_PNG_1, "upload1.png")
        upload_y = await _upload_png(owner, UPLOAD_PNG_2, "upload2.png")
        private = await _upload_png(owner, SAVED_PNG, "private.png")
        admin_file = await _upload_png(t.owui, FOREIGN_PNG, "admin.png")

        # chat X: the owner's first turn, shared with "Allow replies"
        async with BrowserSession(owner) as b:
            await mock.reset()
            cx1 = await b.chat(
                IMAGE_GA,
                "Draw a certificate like this",
                params=LEGACY,
                user_files=[upload],
            )
            cy1 = await b.chat(
                IMAGE_GA,
                "Draw a badge like this final-image-1",
                params=LEGACY,
                user_files=[upload_y],
            )
            cy2 = await b.chat(
                IMAGE_GA,
                "Make it round",
                params=LEGACY,
                chat_id=cy1.chat_id,
                parent_id=cy1.message_id,
            )
        shares = await _share(
            owner,
            cx1.chat_id,
            [_grant("user", part.user["id"]), _grant("group", group_id)],
            replies=True,
        )
        shares += await _share(
            owner, cy1.chat_id, [_grant("user", users["viewer"].user["id"])], False
        )
        folder = await _read_only_folder(owner, cy1.chat_id, users["viewer"].user["id"])
        generated = [
            (f.get("url") or "").split("/files/")[-1].split("/")[0]
            for f in cx1.files
            if f.get("type") == "image"
        ]
        visible = {
            "upload": await _file_status(part, upload["id"]),
            "generated": [await _file_status(part, fid) for fid in generated],
        }

        # B, A and G continue chat X in the browser
        async with BrowserSession(part) as b:
            await mock.reset()
            cb = await b.chat(
                IMAGE_GA,
                f"{EDIT} final-image-2",
                params=LEGACY,
                chat_id=cx1.chat_id,
                parent_id=cx1.message_id,
            )
            rb = await _edit_request(mock)
        async with BrowserSession(owner) as b:
            await mock.reset()
            ca = await b.chat(
                IMAGE_GA,
                "Make it blue final-image-3",
                params=LEGACY,
                chat_id=cx1.chat_id,
                parent_id=cb.message_id,
            )
            ra = await _edit_request(mock)
        async with BrowserSession(users["member"]) as b:
            await mock.reset()
            cg = await b.chat(
                IMAGE_GA,
                "Add a frame",
                params=LEGACY,
                chat_id=cx1.chat_id,
                parent_id=ca.message_id,
            )
            rg = await _edit_request(mock)
        # Open WebUI turns away the Clone-only viewer and the outsider
        async with BrowserSession(users["viewer"]) as b:
            await mock.reset()
            cd = await b.chat(
                IMAGE_GA,
                EDIT,
                params=LEGACY,
                chat_id=cy1.chat_id,
                parent_id=cy2.message_id,
            )
            viewer_calls = len(await mock.requests(_gen))
        async with BrowserSession(users["outsider"]) as b:
            await mock.reset()
            cc = await b.chat(
                IMAGE_GA,
                EDIT,
                params=LEGACY,
                chat_id=cx1.chat_id,
                parent_id=cg.message_id,
            )
            outsider_calls = len(await mock.requests(_gen))

        # the pipe's own check (the filter hands it the chat)
        for key, client, target in (
            ("participant", part, _target(cb)),
            ("admin", t.owui, _target(cb)),
            ("viewer", users["viewer"], _target(cy2)),
            ("outsider", users["outsider"], _target(cb)),
            # the chat check answers True (not the chat) for the admin
            ("truthy", t.owui, f"{_target(cb)} e2e-chat-truthy={cb.chat_id}"),
        ):
            redirected[key] = await _api_edit(client, mock, f"{EDIT} {target}")
        mark = t.mark()
        chat_fault, chat_fault_req = await _api_edit(
            part, mock, f"{EDIT} {_target(cb)} e2e-chat-fault={cb.chat_id}"
        )
        await t.log.settle(0.5)
        chat_fault_logged = len(t.log.lines(mark, "e2e chat access fault"))

        # files named in the message: Open WebUI's rule per file
        url = "/api/v1/files/{}/content"
        mark = t.mark()
        links, links_req = await _api_edit(
            part,
            mock,
            f"Edit these ![a]({url.format(upload['id'])}) "
            f"![b]({url.format(admin_file['id'])}) ![c]({url.format(private['id'])})",
        )
        await t.log.settle(0.5)
        denied = len(t.log.lines(mark, FILE_REFUSED))
        ids_logged = [
            line
            for line in t.log.lines(mark, admin_file["id"], private["id"])
            if "function_gemini" in line
        ]
        mark = t.mark()
        file_fault, file_fault_req = await _api_edit(
            part,
            mock,
            f"Edit this ![a]({url.format(upload['id'])}) e2e-file-fault={upload['id']}",
        )
        await t.log.settle(0.5)
        file_fault_logged = len(
            [
                line
                for line in t.log.lines(mark, "File access check failed")
                if "RuntimeError" in line
            ]
        )
        ids_logged += [
            line
            for line in t.log.lines(mark, upload["id"])
            if "function_gemini" in line
        ]
        t.expect_warnings(mark, ("function_gemini", "File access check failed"))
        await mock.reset()
        video = await part.chat(
            VEO,
            f"Animate this ![frame]({url.format(upload['id'])})",
            stream=False,
            timeout=180,
        )
        video_image = await _video_image(mock)
        sec = await _security(
            t, mock, owner, part, users["outsider"], private, admin_file
        )
    except Exception as exc:  # noqa: BLE001 - reported in the checks
        error = f"setup failed: {exc!r} "
    finally:
        for client in users.values():
            await client.close()
        await t.owui.delete_model(IMAGE_GA)
        await t.owui.delete_model(VEO)
        await t.owui.api("DELETE", f"/api/v1/functions/id/{REDIRECT_FILTER_ID}/delete")
        if group_id:
            await t.owui.api("DELETE", f"/api/v1/groups/id/{group_id}/delete")
    setup = (
        f"{error}grants={grants} shares={shares} group_add={group_status} "
        f"folder={folder} turn1: {_brief(cx1)} files={len(getattr(cx1, 'files', []))}"
    )
    ready = not error and grants == [200, 200, 200, True, True] and shares == [200] * 4

    t.check(
        "shared.visible",
        "Open WebUI lets the participant read the owner's upload and the "
        "generated image of chat X (the pipe links its images to the chat)",
        ready and visible.get("upload") == 200 and visible.get("generated") == [200],
        f"{setup} participant: upload HTTP {visible.get('upload')} "
        f"generated HTTP {visible.get('generated')}",
    )
    t.check(
        "shared.participant",
        "browser, a participant (user grant, Allow replies) edits in the owner's "
        "chat: the owner's upload and the generated image of turn 1 are sent",
        ready
        and getattr(cb, "done", False)
        and rb["sent"] == [f"{EDIT} final-image-2", *WITH_HISTORY[1:]],
        f"{setup} participant: {_brief(cb)} {_edit_report(rb)}",
    )
    t.check(
        "shared.owner",
        "browser, the owner continues after the participant: the participant's "
        "generated image is sent too",
        ready
        and getattr(ca, "done", False)
        and ra["images"] == ["upload1", "final", "gen2"],
        f"{setup} owner: {_brief(ca)} {_edit_report(ra)}",
    )
    t.check(
        "shared.group",
        "browser, a member of a group the chat is shared with edits in it: every "
        "image of the chat is sent",
        ready
        and group_status == 200
        and getattr(cg, "done", False)
        and rg["images"] == ["upload1", "final", "gen2", "gen3"],
        f"{setup} member: {_brief(cg)} {_edit_report(rg)}",
    )
    t.check(
        "shared.refused",
        "browser, a viewer of a Clone only share (also with read-only folder "
        "access) and a user the chat is not shared with: Open WebUI refuses "
        "their turns (HTTP 404), nothing reaches Gemini",
        ready
        and getattr(cd, "http_status", None) == 404
        and getattr(cc, "http_status", None) == 404
        and viewer_calls == 0
        and outsider_calls == 0,
        f"{setup} viewer: {_brief(cd)} upstream={viewer_calls} "
        f"outsider: {_brief(cc)} upstream={outsider_calls}",
    )
    for key, title, expected in (
        (
            "participant",
            "a participant: the chat's images are sent (the filter works)",
            WITH_HISTORY,
        ),
        (
            "admin",
            "an admin (read access to every chat, not write): nothing of the "
            "chat is read",
            [EDIT],
        ),
        (
            "viewer",
            "a viewer with read access only (Clone only share, read-only "
            "folder): nothing of the chat is read",
            [EDIT],
        ),
        (
            "outsider",
            "a user the chat is not shared with: nothing of the chat is read",
            [EDIT],
        ),
        (
            "truthy",
            "the admin, and the chat check answers a truthy value that is not "
            "the chat: nothing of the chat is read",
            [EDIT],
        ),
    ):
        result, req = redirected.get(key, (None, empty))
        t.check(
            f"shared.pipe-{key}",
            "API, the pipe called for a saved chat without Open WebUI's check "
            f"(test filter): {title}",
            ready
            and getattr(result, "status", None) == 200
            and req["sent"] == expected,
            f"{setup} {key}: HTTP {getattr(result, 'status', None)} {_edit_report(req)}",
        )
    t.check(
        "shared.chat-fault",
        "API, the chat access check raises for a participant: the chat is not "
        "read (logged), the answer completes",
        ready
        and getattr(chat_fault, "status", None) == 200
        and chat_fault_req["sent"] == [EDIT]
        and chat_fault_logged > 0,
        f"{setup} HTTP {getattr(chat_fault, 'status', None)} "
        f"{_edit_report(chat_fault_req)} logged={chat_fault_logged}",
    )
    t.check(
        "shared.links",
        "API, markdown links of a participant: the owner's upload of the shared "
        "chat is read, a file of the admin and a file of the owner outside the "
        "chat are not (logged without their ids)",
        ready
        and getattr(links, "status", None) == 200
        and links_req["images"] == ["upload1"]
        and denied == 2
        and not ids_logged,
        f"{setup} HTTP {getattr(links, 'status', None)} {_edit_report(links_req)} "
        f"denied_logged={denied} ids_logged={[short(x, 120) for x in ids_logged]}",
    )
    t.check(
        "shared.file-fault",
        "API, the file access check raises for a participant's link to a file "
        "of the shared chat: the file is not read (logged without its id)",
        ready
        and getattr(file_fault, "status", None) == 200
        and file_fault_req["images"] == []
        and file_fault_logged > 0
        and not ids_logged,
        f"{setup} HTTP {getattr(file_fault, 'status', None)} "
        f"{_edit_report(file_fault_req)} logged={file_fault_logged}",
    )
    t.check(
        "shared.video",
        "Veo image-to-video, API, a participant links the owner's upload of the "
        "shared chat: it is sent as instances[0].image",
        ready and getattr(video, "status", None) == 200 and video_image == "upload1",
        f"{setup} HTTP {getattr(video, 'status', None)} image={video_image!r}",
    )
    sec_ready = ready and not sec["error"] and sec.get("shares") == [200, 200]
    sec_setup = f"{setup} {sec['error']}x2_shares={sec.get('shares')}"
    a3, a4, a5, a7 = sec["a3"], sec["a4"], sec["a5"], sec["a7"]
    t.check(
        "shared.deputy",
        "browser, a participant names a private file of the owner in a message "
        "of the shared chat: the owner's next turn does not send it (logged), the "
        "file stays unreadable for the participant; the owner's own link to it "
        "is sent, in the owner's message and later from the history",
        sec_ready
        and sec.get("before") == 404
        and sec.get("after") == 404
        and sec.get("a3_done")
        and a3.get("images") == ["upload1", "final", "gen2"]
        and sec.get("a3_refused", 0) > 0
        and "saved" in (a4.get("images") or [])
        and sec.get("a5_done")
        and a5.get("images") == ["final", "gen2", "gen3", "saved", "gen1"],
        f"{sec_setup} participant: private HTTP {sec.get('before')} -> "
        f"{sec.get('after')} turn: {sec.get('b2')} owner: images={a3.get('images')} "
        f"refused_logged={sec.get('a3_refused')} own link: {a4.get('images')} "
        f"then {a5.get('images')}",
    )
    t.check(
        "shared.history-cap",
        f"browser, a participant plants {PLANTED} links to files the owner may "
        f"not use: the owner's next turn looks at no more than {HISTORY_CAP} "
        "images of earlier messages (4 x IMAGE_HISTORY_MAX_REFERENCES), logs no "
        "more refusals and completes",
        sec_ready
        and sec.get("a7_done")
        and 0 < sec.get("cap_refused", 0) <= HISTORY_CAP
        and a7.get("images") == ["gen2"],
        f"{sec_setup} plant: {sec.get('b6')} owner: {sec.get('a7_brief')} "
        f"images={a7.get('images')} refused_logged={sec.get('cap_refused')}",
    )
    t.check(
        "shared.link-chat",
        "API, the pipe called for a chat message of another chat (test filter): "
        "a participant's generated image is linked to that chat (the owner may "
        "read it), an outsider's are not (logged; with and without the decision "
        "of the image history)",
        sec_ready
        and sec.get("link_status") == [200, 200, 200]
        and sec.get("owner_reads_b") == [200]
        and sec.get("owner_reads_c") == [404, 404]
        and sec.get("link_refused") == 2,
        f"{sec_setup} HTTP {sec.get('link_status')} owner reads participant's "
        f"{sec.get('owner_reads_b')} outsider's {sec.get('owner_reads_c')} "
        f"refused_logged={sec.get('link_refused')}",
    )

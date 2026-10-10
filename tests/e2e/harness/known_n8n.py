"""Known bugs of pipelines/n8n/n8n.py and pipelines/infomaniak/infomaniak.py
(see ``harness.known``).

None is registered right now: the last one, ``infomaniak-name-prefix`` (the
NAME_PREFIX valve was read only once), was fixed in infomaniak.py 2.2.2 and its
check ``infomaniak.models.name-prefix`` must pass. A new entry imports
``FOUND_BY_E2E`` / ``KnownIssue`` from ``.known``.
"""

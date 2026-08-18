"""
Knowledge base seed data. These are hand-written because the Kaggle
dataset's Resolution field is sparse (see data/clean_data.py's fill-rate
). Covers all 7 categories with 3-4 entries each.

the original 25 entries were phrased to closely match the
hand-written eval tickets, and scored well against them (90% hit
rate). Running data/diagnose_retrieval_scores.py against the 50 REAL
Kaggle-sourced tickets in eval_tune.csv exposed a generalization gap --
every single ticket scored below the 0.55 similarity threshold (range
0.19-0.52), including cases that were clearly the right category (e.g.
"Network problem" scored 0.505 against "Data disappeared after a failed
sync" -- directionally right, not textually close enough).

The entries below target that gap: phrased to reflect the *patterns* seen
in eval_tune.csv's real tickets (firmware-triggered issues, factory-reset-
didn't-help, peripheral/charging problems, vague security concerns,
multiple-failed-support-contacts, etc.), written in original wording --
deliberately NOT copied or closely paraphrased from the eval tickets
themselves, since retrieval tested against a KB built from the same
tickets it's evaluated on would be circular and would invalidate the eval

 re-running diagnose_retrieval_scores.py against the same
50-ticket tune set after the above expansion (45 entries) still showed a
~24% match rate (12/50). A follow-up gap analysis on the 38 remaining
misses -- cross-checking actual ticket descriptions, not just subjects,
since the Kaggle subjects are frequently mismatched to the real complaint
-- surfaced further concrete gaps. Five more entries were added below to
target those (see README Results for before/after numbers). One candidate
entry ("same bug reported by multiple users on the same device model") was
dropped from this batch as a near-duplicate of an existing Software entry
below, rather than kept as redundant KB content that splits similarity
mass across two near-identical entries.
"""

KNOWLEDGE_BASE_SEED = [
    # Network
    {"category": "Network", "issue_summary": "WiFi keeps disconnecting intermittently",
     "resolution_text": "Restart the router, forget and rejoin the network, and update WiFi adapter drivers. If issue persists on one device only, it's likely a driver issue; if it affects all devices, it's the router."},
    {"category": "Network", "issue_summary": "Cannot connect to VPN from home network",
     "resolution_text": "Confirm VPN client is updated to latest version. Check firewall isn't blocking VPN ports (usually UDP 500/4500). Try switching from WiFi to wired connection to rule out network instability."},
    {"category": "Network", "issue_summary": "Video calls drop repeatedly during meetings",
     "resolution_text": "Run a speed test during the drop window to check for bandwidth issues. Switch to a 5GHz WiFi band if on 2.4GHz. Close bandwidth-heavy background apps (cloud sync, streaming)."},
    {"category": "Network", "issue_summary": "Slow internet speed compared to plan",
     "resolution_text": "Run speed test wired directly to modem to isolate WiFi vs ISP issue. Restart modem and router. If wired speed also low, contact ISP with test results."},

    # Account Access
    {"category": "Account Access", "issue_summary": "Forgot password and reset email not arriving",
     "resolution_text": "Check spam/junk folder first. Confirm the email on file is correct. If still not received after 10 minutes, manually trigger a password reset from the admin panel."},
    {"category": "Account Access", "issue_summary": "Account locked after multiple failed login attempts",
     "resolution_text": "Lockouts auto-clear after 24 hours. To unlock immediately, verify user identity via backup email/phone, then manually reset the lockout flag in the admin panel."},
    {"category": "Account Access", "issue_summary": "Two-factor authentication code never arrives",
     "resolution_text": "Confirm phone number on file is correct and has signal. Offer backup codes or email-based 2FA as an alternative. Check SMS provider status page for outages."},
    {"category": "Account Access", "issue_summary": "Cannot log in on new device",
     "resolution_text": "New-device logins sometimes trigger extra verification. Confirm user has access to their verification email/phone. Clear browser cache/cookies if using a browser previously logged into another account."},

    # Hardware
    {"category": "Hardware", "issue_summary": "Laptop overheating during normal use",
     "resolution_text": "Check for dust buildup in vents (recommend compressed air cleaning). Verify background processes aren't maxing CPU. Ensure laptop is on a hard, flat surface for airflow."},
    {"category": "Hardware", "issue_summary": "Printer not detected by computer",
     "resolution_text": "Restart both printer and computer. Reinstall or update printer drivers. Confirm printer and computer are on the same network if using WiFi printing."},
    {"category": "Hardware", "issue_summary": "Battery draining much faster than usual",
     "resolution_text": "Check battery health in system settings — if below 80% capacity, battery replacement may be needed. Otherwise, check for a runaway background process draining power."},
    {"category": "Hardware", "issue_summary": "External monitor not detected after driver update",
     "resolution_text": "Roll back to the previous graphics driver version if the update caused the issue. Try a different cable/port to rule out a hardware fault. Re-detect displays in display settings."},

    # Software
    {"category": "Software", "issue_summary": "Application crashes immediately on startup",
     "resolution_text": "Clear the application's cache/config folder and relaunch. Reinstall the application if the issue persists. Check for a pending OS update that may cause compatibility issues."},
    {"category": "Software", "issue_summary": "Software update installer gets stuck partway through",
     "resolution_text": "Cancel and restart the installer. Ensure at least 2GB free disk space. Temporarily disable antivirus during install, as it can interfere with installer processes."},
    {"category": "Software", "issue_summary": "Feature that used to work now throws an error",
     "resolution_text": "Check release notes for the most recent update for known issues with that feature. Try clearing local cache. Escalate to engineering if error persists after cache clear with exact error message."},

    # Billing
    {"category": "Billing", "issue_summary": "Charged twice for the same subscription period",
     "resolution_text": "Check payment processor logs for duplicate transaction IDs. If confirmed duplicate, issue refund for the extra charge and note the payment gateway ticket ID for tracking."},
    {"category": "Billing", "issue_summary": "Refund requested but not received after two weeks",
     "resolution_text": "Refunds typically take 5-10 business days depending on the bank. Verify refund was actually processed on our end (check payment gateway dashboard). If confirmed processed, advise user to check with their bank."},
    {"category": "Billing", "issue_summary": "Invoice shows incorrect tax amount",
     "resolution_text": "Verify the tax rate configured for the user's billing region against current rates. If a rate table error is found, correct it and reissue a corrected invoice."},
    {"category": "Billing", "issue_summary": "Charged for a plan tier the user did not select",
     "resolution_text": "Check account's plan-change history for when the upgrade occurred and whether it was user-initiated or automatic (e.g. usage-based tier change). Downgrade and refund the difference if it was unintended."},

    # Data Loss
    {"category": "Data Loss", "issue_summary": "Lost unsaved file after application crash",
     "resolution_text": "Check the application's auto-recovery folder (most apps save a recovery copy every few minutes). If auto-save was disabled, unfortunately the file may not be recoverable."},
    {"category": "Data Loss", "issue_summary": "Accidentally deleted an important email",
     "resolution_text": "Check the Trash/Deleted Items folder first — most email systems retain deleted items for 30 days. If already permanently deleted, check if the account has a backup/archive feature enabled."},
    {"category": "Data Loss", "issue_summary": "Data disappeared after a failed sync",
     "resolution_text": "Do not sync again until investigated — check for a local cache/backup copy before it gets overwritten. Review sync logs for the point of failure. Restore from the most recent successful sync point if available."},

    # General Inquiry
    {"category": "General Inquiry", "issue_summary": "How to upgrade from basic to premium plan",
     "resolution_text": "Direct the user to Account Settings > Subscription > Change Plan. No troubleshooting needed — this is a self-service action. Offer to walk through it live if they're unable to find the option."},
    {"category": "General Inquiry", "issue_summary": "Feature request for a capability that doesn't exist yet",
     "resolution_text": "Thank the user for the suggestion and log it in the feature request tracker. No fix needed — this is not a bug. Provide an estimated timeline only if one is publicly available."},
    {"category": "General Inquiry", "issue_summary": "General question about how a feature works",
     "resolution_text": "Point to the relevant help center article. If none exists, explain the feature directly and flag the documentation gap internally."},

    # --- Week 3 additions (see module docstring) ---

    # Network
    {"category": "Network", "issue_summary": "Device won't connect to any WiFi network during initial setup",
     "resolution_text": "Confirm the device supports the router's band (some devices are 2.4GHz-only). Re-enter the WiFi password carefully (case-sensitive). Try connecting to a different network to isolate a device fault from a router-specific issue. If nothing connects, factory reset the device's network settings and retry setup from scratch."},
    {"category": "Network", "issue_summary": "Smart home or IoT device disconnects from WiFi randomly throughout the day",
     "resolution_text": "Check the router's DHCP lease time -- short leases can cause smart devices to silently drop off. Move the device closer to the router temporarily to rule out signal strength. Update the device's firmware, since many IoT WiFi bugs are fixed in later firmware versions rather than router-side changes."},
    {"category": "Network", "issue_summary": "User is worried about the security of their network-connected device and whether their data is safe",
     "resolution_text": "Confirm the device's firmware is up to date, since outdated firmware is the most common real security gap. Walk the user through checking connected-device/session lists in the account or router admin panel for anything unrecognized. Recommend a unique WiFi password if the network password is shared/weak."},

    # Account Access
    {"category": "Account Access", "issue_summary": "Login fails with an invalid credentials error even though the password looks correct",
     "resolution_text": "Have the user retype the password manually rather than using autofill, since saved/cached passwords are a common silent mismatch. Confirm Caps Lock isn't on. If it still fails, trigger a manual password reset even though the password 'looks' right -- the stored hash may be out of sync after a past reset attempt."},
    {"category": "Account Access", "issue_summary": "User suspects unauthorized access to their account and wants to secure it",
     "resolution_text": "Review the account's recent login/session activity with the user and identify anything unrecognized. Force-logout all active sessions except the current one. Walk them through resetting their password and enabling two-factor authentication as a precaution, even if no confirmed breach is found."},
    {"category": "Account Access", "issue_summary": "Login works sometimes but fails unpredictably other times",
     "resolution_text": "Check for session token expiration or concurrent-login conflicts (e.g. the same account logged in on another device invalidating the session). Have the user clear cookies/cache for the login page. If using SSO, confirm the identity provider's session timeout isn't shorter than expected."},

    # Hardware
    {"category": "Hardware", "issue_summary": "Device won't charge properly even with the original charger and cable",
     "resolution_text": "Test with a different wall outlet/power source to rule out the outlet itself. Inspect the charging port for visible debris or damage. If the original cable/charger still fails on a known-good outlet, this points to a genuine hardware fault in the port or battery, not a cable issue -- route to warranty/repair."},
    {"category": "Hardware", "issue_summary": "Problem started right after a firmware update was installed on the device",
     "resolution_text": "Check the firmware release notes for known regressions in this version. If a rollback path exists, walk the user through reverting to the previous firmware version. If no rollback is available, escalate to the hardware/firmware team with the exact version number and symptom."},
    {"category": "Hardware", "issue_summary": "Issue persists even after performing a full factory reset on the device",
     "resolution_text": "A factory reset only resolves software-state problems, so a symptom that survives a reset strongly suggests a genuine hardware fault rather than a configuration issue. Don't repeat further reset/reconfiguration steps -- route directly to warranty/repair with a note that reset was already attempted and did not help."},
    {"category": "Hardware", "issue_summary": "An accessory or peripheral isn't recognized when connected to the main device",
     "resolution_text": "Confirm the accessory's firmware/model is listed as compatible with this specific device model, since not all accessories in a product line are universally compatible. Try a different cable and port to rule out a connection-specific fault before assuming an incompatibility."},

    # Software
    {"category": "Software", "issue_summary": "App or feature crashes only when performing one specific action, not generally",
     "resolution_text": "Reproduce the exact steps the user describes rather than general app usage, since the bug is likely isolated to that specific code path. Check crash logs filtered to that action. If not already fixed in the latest patch, route to engineering with the precise reproduction steps rather than a general 'app crashes' report."},
    {"category": "Software", "issue_summary": "User confirms they're already on the latest software version but the issue remains",
     "resolution_text": "Since updating isn't the fix here, shift focus to local cache/data corruption instead -- walk the user through clearing the app's cache or local data (not a full reinstall yet). If that doesn't help, this may be a bug not yet fixed in the current release; log it for engineering with the exact version number."},
    {"category": "Software", "issue_summary": "Same bug is being reported by multiple users on the same device model",
     "resolution_text": "This pattern indicates a systemic defect rather than a one-off, device-specific issue. Skip individual troubleshooting steps and escalate directly to engineering as a broader known-issue report, referencing the other affected tickets if available."},
    {"category": "Software", "issue_summary": "App freezes frequently and becomes unresponsive during normal use",
     "resolution_text": "Check whether resource usage (memory/CPU) climbs steadily before each freeze, which points to a memory leak. Clear the app's cache and confirm the device has adequate free storage. If freezes correlate with a specific feature, note that pattern for engineering rather than reporting it as a general freeze."},

    # Data Loss
    {"category": "Data Loss", "issue_summary": "All files disappeared after a factory reset or full device wipe",
     "resolution_text": "Since a reset/wipe was performed, the primary recovery path is a cloud backup made before the reset -- check backup settings and history for the most recent successful backup and restore from there. Local recovery after a full wipe is usually not possible."},
    {"category": "Data Loss", "issue_summary": "Device crashed and all previously stored data now appears to be gone",
     "resolution_text": "Advise the user to stop using the device for further writes immediately, since continued use risks overwriting recoverable data. Check for any cloud sync or backup enabled on the account first. If none exists and the data is critical, recommend a professional data recovery service rather than DIY recovery attempts."},

    # Billing
    {"category": "Billing", "issue_summary": "User reports an ongoing billing problem that's affecting their ability to keep using the product, without specifying the exact issue",
     "resolution_text": "Since no specific billing detail was given, don't guess -- ask the user for the exact charge date, amount, and what they expected instead. Escalate for a manual account review rather than attempting a generic billing fix based on assumptions."},
    {"category": "Billing", "issue_summary": "User has already contacted support multiple times about the same unresolved billing issue",
     "resolution_text": "Do not restart standard troubleshooting, since it has already failed multiple times for this user. Escalate directly to a senior billing agent with a note referencing the repeat contacts, and prioritize a same-day resolution given the prior friction."},

    # General Inquiry
    {"category": "General Inquiry", "issue_summary": "User wants to cancel or downgrade but the ticket doesn't specify enough detail to act on",
     "resolution_text": "Ask the user to confirm exactly what they want cancelled or downgraded and the effective date they want it applied. Do not process a cancellation/downgrade based on assumption -- route to the account/billing team once details are confirmed."},
    {"category": "General Inquiry", "issue_summary": "User is asking for a product or feature recommendation rather than reporting a problem",
     "resolution_text": "This is a pre-purchase or advisory request, not a bug or fault. Point to the relevant product comparison page, or ask a clarifying question about their use case before recommending a specific option."},

    # --- Week 4 additions (see module docstring) ---

    # Hardware
    {"category": "Hardware", "issue_summary": "Battery life has been gradually declining over weeks or months of normal use",
     "resolution_text": "Gradual battery capacity loss over time is usually normal wear rather than a defect — most rechargeable batteries lose some capacity after repeated charge cycles. Try recalibrating the battery by fully draining it and then charging it to 100% without interruption, and check for a software update that may include power management improvements. If the device is still within its warranty period and battery life has dropped significantly (more than 20% below original), a hardware replacement may be warranted."},

    # Network
    {"category": "Network", "issue_summary": "Device fails to detect or connect to any WiFi network right after first unboxing or initial setup",
     "resolution_text": "During first-time setup, make sure the device's WiFi radio is enabled and you're within range of the router. Try forgetting the network on the device and re-entering the WiFi password carefully, especially for networks with special characters. If the device still won't detect any networks at all (not just one specific network), a factory reset before retrying setup often resolves first-boot connectivity issues."},

    # General Inquiry
    {"category": "General Inquiry", "issue_summary": "User can't locate a specific feature, setting, or option and needs step-by-step navigation guidance",
     "resolution_text": "This is usually a discoverability issue rather than a bug — the feature likely exists but isn't where the user expects. Point them to the specific menu path (Settings > [relevant section]) for the feature they're looking for, and mention that a search bar within Settings (if available) can help locate options faster than manual navigation."},

    # Software
    {"category": "Software", "issue_summary": "Device or app displays a vague or unlabeled error message with no clear explanation of what went wrong",
     "resolution_text": "Ask the user for the exact error code or message text if not already provided, since generic messages like 'an unspecified error' usually map to a specific internal error code in the device or app's logs. In the meantime, a restart of the device or app often clears transient unspecified errors. If the message persists after a restart, check for a pending software update, since vague error messages are sometimes a symptom of an outdated app version encountering an unhandled edge case."},

    # Account Access
    {"category": "Account Access", "issue_summary": "User expresses general worry about whether their personal data is safe or secure on a connected device, without describing a specific breach or incident",
     "resolution_text": "This is a proactive security concern rather than a report of an actual incident, so reassure the user and walk them through good baseline practices: confirm their device firmware and companion app are on the latest version, enable two-factor authentication if the account supports it, and review which third-party apps or services have access to their account data. If they have a specific reason for concern (e.g. a suspicious login notification), ask for those details separately, since that would warrant a different, more urgent response."},
]
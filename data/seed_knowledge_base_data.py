"""
Knowledge base seed data. These are hand-written because the Kaggle
dataset's Resolution field is sparse (see data/clean_data.py's fill-rate
check from Week 1). Covers all 7 categories with 3-4 entries each.
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
]

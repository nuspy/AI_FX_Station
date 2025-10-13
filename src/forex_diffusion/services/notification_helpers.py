"""
Notification helpers for circuit breaker alerts.

Provides optional integrations for:
- Slack notifications
- Email alerts
- Webhook calls
- SMS (Twilio)

Usage:
    from .notification_helpers import send_slack_alert
    
    def on_circuit_open(cb):
        send_slack_alert(
            f"🚨 {cb.service_name} circuit breaker opened!",
            severity="critical"
        )
"""
from __future__ import annotations

import os
from typing import Optional, Dict, Any
from loguru import logger


def send_slack_alert(
    message: str,
    severity: str = "warning",
    webhook_url: Optional[str] = None,
    channel: Optional[str] = None
) -> bool:
    """
    Send alert to Slack via webhook.
    
    Args:
        message: Alert message to send
        severity: Severity level ("info", "warning", "critical")
        webhook_url: Slack webhook URL (default: from SLACK_WEBHOOK_URL env)
        channel: Channel to post to (default: webhook default channel)
    
    Returns:
        True if sent successfully, False otherwise
        
    Environment Variables:
        SLACK_WEBHOOK_URL: Default Slack webhook URL
        
    Example:
        send_slack_alert(
            "🚨 AggregatorService circuit breaker opened!",
            severity="critical"
        )
    """
    try:
        import httpx
        
        webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            logger.debug("Slack webhook URL not configured, skipping alert")
            return False
        
        # Color-code by severity
        colors = {
            "info": "#36a64f",      # Green
            "warning": "#ff9900",   # Orange
            "critical": "#ff0000"   # Red
        }
        color = colors.get(severity, "#808080")
        
        # Emoji by severity
        emojis = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨"
        }
        emoji = emojis.get(severity, "📢")
        
        payload: Dict[str, Any] = {
            "attachments": [{
                "color": color,
                "text": f"{emoji} {message}",
                "footer": "ForexGPT Service Monitor",
                "ts": int(__import__("time").time())
            }]
        }
        
        if channel:
            payload["channel"] = channel
        
        response = httpx.post(webhook_url, json=payload, timeout=5.0)
        response.raise_for_status()
        
        logger.debug(f"Slack alert sent: {message[:50]}...")
        return True
        
    except ImportError:
        logger.warning("httpx not installed, cannot send Slack alerts")
        return False
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")
        return False


def send_email_alert(
    subject: str,
    message: str,
    to_email: Optional[str] = None,
    smtp_host: Optional[str] = None,
    smtp_port: int = 587,
    smtp_user: Optional[str] = None,
    smtp_password: Optional[str] = None
) -> bool:
    """
    Send alert via email (SMTP).
    
    Args:
        subject: Email subject
        message: Email body (plain text)
        to_email: Recipient email (default: from ALERT_EMAIL env)
        smtp_host: SMTP server (default: from SMTP_HOST env)
        smtp_port: SMTP port (default: 587 for TLS)
        smtp_user: SMTP username (default: from SMTP_USER env)
        smtp_password: SMTP password (default: from SMTP_PASSWORD env)
    
    Returns:
        True if sent successfully, False otherwise
        
    Environment Variables:
        ALERT_EMAIL: Default recipient email
        SMTP_HOST: SMTP server hostname
        SMTP_USER: SMTP username
        SMTP_PASSWORD: SMTP password
        
    Example:
        send_email_alert(
            "Circuit Breaker Alert",
            "AggregatorService circuit breaker opened after 5 failures."
        )
    """
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # Get config from env if not provided
        to_email = to_email or os.getenv("ALERT_EMAIL")
        smtp_host = smtp_host or os.getenv("SMTP_HOST")
        smtp_user = smtp_user or os.getenv("SMTP_USER")
        smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        
        if not all([to_email, smtp_host, smtp_user, smtp_password]):
            logger.debug("Email config incomplete, skipping email alert")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Subject"] = f"[ForexGPT Alert] {subject}"
        
        body = MIMEText(message, "plain")
        msg.attach(body)
        
        # Send via SMTP
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logger.debug(f"Email alert sent to {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")
        return False


def send_webhook_alert(
    message: str,
    webhook_url: Optional[str] = None,
    severity: str = "warning",
    extra_data: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Send alert to generic webhook endpoint.
    
    Args:
        message: Alert message
        webhook_url: Webhook URL (default: from ALERT_WEBHOOK_URL env)
        severity: Severity level
        extra_data: Additional data to include in payload
    
    Returns:
        True if sent successfully, False otherwise
        
    Environment Variables:
        ALERT_WEBHOOK_URL: Default webhook URL
        
    Example:
        send_webhook_alert(
            "Circuit breaker opened",
            extra_data={"service": "AggregatorService", "failures": 5}
        )
    """
    try:
        import httpx
        
        webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL")
        if not webhook_url:
            logger.debug("Webhook URL not configured, skipping alert")
            return False
        
        payload = {
            "message": message,
            "severity": severity,
            "timestamp": __import__("time").time(),
            "source": "forexgpt"
        }
        
        if extra_data:
            payload.update(extra_data)
        
        response = httpx.post(webhook_url, json=payload, timeout=5.0)
        response.raise_for_status()
        
        logger.debug(f"Webhook alert sent: {message[:50]}...")
        return True
        
    except ImportError:
        logger.warning("httpx not installed, cannot send webhook alerts")
        return False
    except Exception as e:
        logger.error(f"Failed to send webhook alert: {e}")
        return False


def create_slack_notifier(webhook_url: Optional[str] = None):
    """
    Create a circuit breaker notification callback for Slack.
    
    Args:
        webhook_url: Slack webhook URL (default: from env)
    
    Returns:
        Callable that can be used as on_open/on_close callback
        
    Example:
        from .notification_helpers import create_slack_notifier
        
        service = AggregatorService(
            engine,
            ...
        )
        
        # Override circuit breaker callbacks
        notifier_open = create_slack_notifier()
        service._circuit_breaker.on_open = notifier_open
    """
    def notify(circuit_breaker):
        if circuit_breaker.state.value == "open":
            send_slack_alert(
                f"🚨 {circuit_breaker.service_name} circuit breaker OPENED "
                f"after {circuit_breaker._failure_count} failures",
                severity="critical",
                webhook_url=webhook_url
            )
        elif circuit_breaker.state.value == "closed":
            send_slack_alert(
                f"✅ {circuit_breaker.service_name} circuit breaker CLOSED "
                f"(service recovered)",
                severity="info",
                webhook_url=webhook_url
            )
    
    return notify


def create_email_notifier(to_email: Optional[str] = None):
    """
    Create a circuit breaker notification callback for email.
    
    Args:
        to_email: Recipient email (default: from env)
    
    Returns:
        Callable that can be used as on_open/on_close callback
        
    Example:
        from .notification_helpers import create_email_notifier
        
        service = AggregatorService(engine)
        notifier = create_email_notifier("admin@example.com")
        service._circuit_breaker.on_open = notifier
    """
    def notify(circuit_breaker):
        if circuit_breaker.state.value == "open":
            send_email_alert(
                f"{circuit_breaker.service_name} Circuit Breaker Opened",
                f"Service: {circuit_breaker.service_name}\n"
                f"Status: Circuit breaker OPENED\n"
                f"Failures: {circuit_breaker._failure_count} consecutive\n"
                f"Timeout: {circuit_breaker.timeout}s\n\n"
                f"The service will be blocked until the timeout expires and recovery is confirmed.",
                to_email=to_email
            )
        elif circuit_breaker.state.value == "closed":
            send_email_alert(
                f"{circuit_breaker.service_name} Circuit Breaker Closed",
                f"Service: {circuit_breaker.service_name}\n"
                f"Status: Circuit breaker CLOSED (recovered)\n"
                f"Successful operations: {circuit_breaker._success_count}\n\n"
                f"The service has recovered and is operating normally.",
                to_email=to_email
            )
    
    return notify

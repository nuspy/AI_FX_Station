"""
Credentials Manager - Secure storage using OS keyring + Fernet encryption.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from loguru import logger

try:
    import keyring
    from cryptography.fernet import Fernet
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False
    logger.warning("keyring or cryptography not installed. Credentials will not be encrypted.")


@dataclass
class ProviderCredentials:
    """Credentials for a market data provider."""

    provider_name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    environment: str = "demo"  # "demo" or "live"
    metadata: Dict[str, Any] = None  # Additional provider-specific data

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProviderCredentials:
        """Create from dictionary."""
        return cls(**data)


class CredentialsManager:
    """
    Manages provider credentials with secure storage.

    Uses OS-level keyring (Windows Credential Manager, macOS Keychain, etc.)
    with Fernet encryption for additional security.
    """

    KEYRING_SERVICE = "ForexGPT"
    ENCRYPTION_KEY_NAME = "encryption_key"

    def __init__(self):
        if not _HAS_CRYPTO:
            raise ImportError(
                "Required packages not installed. "
                "Install with: pip install keyring cryptography"
            )

        self._encryption_key: Optional[bytes] = None
        self._ensure_encryption_key()

    def _ensure_encryption_key(self) -> None:
        """Ensure encryption key exists in keyring."""
        try:
            # Try to get existing key
            key_str = keyring.get_password(self.KEYRING_SERVICE, self.ENCRYPTION_KEY_NAME)

            if key_str:
                self._encryption_key = key_str.encode()
            else:
                # Generate new key
                self._encryption_key = Fernet.generate_key()
                keyring.set_password(
                    self.KEYRING_SERVICE,
                    self.ENCRYPTION_KEY_NAME,
                    self._encryption_key.decode()
                )
                logger.info("Generated new encryption key in keyring")

        except Exception as e:
            logger.error(f"Failed to setup encryption key: {e}")
            raise

    def _encrypt(self, data: str) -> str:
        """Encrypt data using Fernet."""
        if not self._encryption_key:
            raise RuntimeError("Encryption key not initialized")

        fernet = Fernet(self._encryption_key)
        encrypted = fernet.encrypt(data.encode())
        return encrypted.decode()

    def _decrypt(self, encrypted_data: str) -> str:
        """Decrypt data using Fernet."""
        if not self._encryption_key:
            raise RuntimeError("Encryption key not initialized")

        fernet = Fernet(self._encryption_key)
        decrypted = fernet.decrypt(encrypted_data.encode())
        return decrypted.decode()

    def save(self, credentials: ProviderCredentials) -> None:
        """
        Save credentials to keyring with encryption.

        Args:
            credentials: Provider credentials to save
        """
        try:
            # Convert to JSON
            creds_dict = credentials.to_dict()
            creds_json = json.dumps(creds_dict)

            # Encrypt
            encrypted = self._encrypt(creds_json)

            # Store in keyring
            keyring.set_password(
                self.KEYRING_SERVICE,
                credentials.provider_name,
                encrypted
            )

            logger.info(f"Saved credentials for provider: {credentials.provider_name}")

        except Exception as e:
            logger.error(f"Failed to save credentials for {credentials.provider_name}: {e}")
            raise

    def load(self, provider_name: str) -> Optional[ProviderCredentials]:
        """
        Load credentials from keyring.

        Args:
            provider_name: Name of provider

        Returns:
            ProviderCredentials if found, None otherwise
        """
        try:
            # Get from keyring
            encrypted = keyring.get_password(self.KEYRING_SERVICE, provider_name)

            if not encrypted:
                logger.debug(f"No credentials found for provider: {provider_name}")
                return None

            # Decrypt
            creds_json = self._decrypt(encrypted)

            # Parse JSON
            creds_dict = json.loads(creds_json)

            # Create credentials object
            credentials = ProviderCredentials.from_dict(creds_dict)

            logger.debug(f"Loaded credentials for provider: {provider_name}")
            return credentials

        except Exception as e:
            logger.error(f"Failed to load credentials for {provider_name}: {e}")
            return None

    def delete(self, provider_name: str) -> None:
        """
        Delete credentials from keyring.

        Args:
            provider_name: Name of provider
        """
        try:
            keyring.delete_password(self.KEYRING_SERVICE, provider_name)
            logger.info(f"Deleted credentials for provider: {provider_name}")

        except keyring.errors.PasswordDeleteError:
            logger.warning(f"No credentials to delete for provider: {provider_name}")

        except Exception as e:
            logger.error(f"Failed to delete credentials for {provider_name}: {e}")
            raise

    def list_providers(self) -> List[str]:
        """
        List all providers with stored credentials.

        Returns:
            List of provider names

        Note: This method has limited support due to keyring limitations.
        It attempts to check common provider names.
        """
        common_providers = ["tiingo", "ctrader", "alphavantage"]
        found_providers = []

        for provider in common_providers:
            if keyring.get_password(self.KEYRING_SERVICE, provider):
                found_providers.append(provider)

        return found_providers

    def update_token(self, provider_name: str, access_token: str, refresh_token: Optional[str] = None) -> None:
        """
        Update access/refresh tokens for a provider.

        Args:
            provider_name: Name of provider
            access_token: New access token
            refresh_token: New refresh token (optional)
        """
        credentials = self.load(provider_name)

        if not credentials:
            logger.error(f"Cannot update token: credentials not found for {provider_name}")
            raise ValueError(f"Credentials not found for {provider_name}")

        # Update tokens
        credentials.access_token = access_token

        if refresh_token:
            credentials.refresh_token = refresh_token

        # Save
        self.save(credentials)
        logger.info(f"Updated tokens for provider: {provider_name}")


# Global instance
_global_manager: Optional[CredentialsManager] = None


def get_credentials_manager() -> CredentialsManager:
    """Get global CredentialsManager instance."""
    global _global_manager

    if _global_manager is None:
        _global_manager = CredentialsManager()

    return _global_manager

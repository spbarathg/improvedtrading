import os
import logging
from solana.keypair import Keypair
from base64 import b64decode

class KeyManager:
    def __init__(self):
        self.private_key = None

    def load_private_key(self):
        """
        Loads the private key from an environment variable, decodes it, and stores it securely.
        The environment variable should be in base64 encoded format for security.
        """
        try:
            private_key_b64 = os.getenv("PRIVATE_KEY_B64")
            if not private_key_b64:
                raise ValueError("Private key environment variable not set.")

            # Decode the base64 encoded private key
            private_key_bytes = b64decode(private_key_b64)
            
            # Ensure the key length is correct (Solana Keypair expects 64 bytes)
            if len(private_key_bytes) != 64:
                raise ValueError("Invalid private key length.")
            
            # Create a Solana Keypair object from the decoded bytes
            self.private_key = Keypair.from_secret_key(private_key_bytes)
            
            logging.info("Private key successfully loaded.")
        except Exception as e:
            logging.error(f"Error loading private key: {e}")
            raise

    def get_keypair(self):
        """
        Returns the Keypair object after loading the private key.
        If the private key is not loaded, it calls the load_private_key method.
        Returns:
            Keypair: Solana Keypair object with the bot's private key.
        """
        if self.private_key is None:
            self.load_private_key()
        return self.private_key

    def get_public_key(self):
        """
        Returns the public key associated with the private key.
        Returns:
            PublicKey: Solana public key.
        """
        if self.private_key is None:
            self.load_private_key()
        return self.private_key.public_key
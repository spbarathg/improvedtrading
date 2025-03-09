import aiohttp
import base64
import json
import logging
import asyncio
from solana.rpc.async_api import AsyncClient
from solana.transaction import Transaction
from solana.publickey import PublicKey
from solana.system_program import TransferParams, transfer
from solana.keypair import Keypair
from solana.rpc.types import TxOpts
from trading_bot.utils.key_manager import KeyManager  # KeyManager handles loading the private key securely

class Exchange:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger()
        self.key_manager = KeyManager(self.config)  # Securely load private key
        self.private_key = self.key_manager.get_private_key()
        self.client = AsyncClient(self.config.solana_rpc_url)

    async def get_swap_quote(self, input_token, output_token, amount):
        """Get token swap quote from Jupiter."""
        url = f"https://quote-api.jup.ag/v4/quote?inputMint={input_token}&outputMint={output_token}&amount={amount}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        self.logger.warning(f"Failed to get swap quote, status code: {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error fetching swap quote: {e}")
            return None

    async def build_transaction(self, sender_pubkey, receiver_pubkey, lamports):
        """Build a Solana transaction for transfer."""
        try:
            params = TransferParams(
                from_pubkey=PublicKey(sender_pubkey),
                to_pubkey=PublicKey(receiver_pubkey),
                lamports=lamports
            )
            tx = Transaction().add(transfer(params))
            return tx
        except Exception as e:
            self.logger.error(f"Error building transaction: {e}")
            return None

    async def sign_transaction(self, transaction):
        """Sign a Solana transaction using the bot's private key."""
        try:
            # Sign the transaction
            signer = Keypair.from_secret_key(base64.b64decode(self.private_key))
            transaction.sign(signer)
            return transaction
        except Exception as e:
            self.logger.error(f"Error signing transaction: {e}")
            return None

    async def send_transaction(self, transaction):
        """Send a Solana transaction and return the result."""
        try:
            response = await self.client.send_transaction(transaction, opts=TxOpts(skip_confirmation=False))
            return response
        except Exception as e:
            self.logger.error(f"Error sending transaction: {e}")
            return None

    async def confirm_transaction(self, signature):
        """Confirm transaction via Solana websocket (accountSubscribe)."""
        try:
            async with self.client.ws_subscribe('accountSubscribe', {'account': signature}) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get('result'):
                            self.logger.info(f"Transaction {signature} confirmed")
                            return True
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        self.logger.error(f"Websocket error: {msg.data}")
                        break
        except Exception as e:
            self.logger.error(f"Error confirming transaction: {e}")
            return False

    async def execute_trade(self, input_token, output_token, amount):
        """Main method to execute a token swap via Jupiter."""
        # Get swap quote
        quote = await self.get_swap_quote(input_token, output_token, amount)
        if not quote:
            self.logger.error(f"Failed to retrieve quote for {input_token} -> {output_token}")
            return

        # Build transaction based on the quote (placeholder code for building swap transaction)
        transaction = await self.build_transaction(self.config.sender_pubkey, self.config.receiver_pubkey, amount)
        if not transaction:
            self.logger.error("Failed to build transaction.")
            return

        # Sign transaction
        signed_tx = await self.sign_transaction(transaction)
        if not signed_tx:
            self.logger.error("Failed to sign transaction.")
            return

        # Send transaction
        response = await self.send_transaction(signed_tx)
        if not response or not response.get('result'):
            self.logger.error("Failed to send transaction.")
            return

        # Confirm transaction
        signature = response['result']
        confirmed = await self.confirm_transaction(signature)
        if confirmed:
            self.logger.info(f"Swap {input_token} -> {output_token} executed successfully!")
        else:
            self.logger.error(f"Transaction {signature} failed to confirm.")

    async def close(self):
        """Close the Solana RPC client."""
        await self.client.close()
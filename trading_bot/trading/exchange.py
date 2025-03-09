import aiohttp
import base64
import json
import logging
import asyncio
from solana.rpc.async_api import AsyncClient
from solana.transaction import Transaction
from solana.publickey import PublicKey
from solana.keypair import Keypair
from solana.rpc.types import TxOpts
from trading_bot.utils.key_manager import KeyManager

class Exchange:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger()
        self.key_manager = KeyManager(self.config)
        self.private_key = self.key_manager.get_private_key()
        self.client = AsyncClient(self.config.solana_rpc_url)
        self.jupiter_api_url = "https://quote-api.jup.ag/v4"

    async def get_swap_quote(self, input_token, output_token, amount):
        """Get token swap quote from Jupiter."""
        url = f"{self.jupiter_api_url}/quote?inputMint={input_token}&outputMint={output_token}&amount={amount}&slippageBps=50"
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

    async def get_swap_transaction(self, quote_response):
        """Get swap transaction from Jupiter using quote response."""
        url = f"{self.jupiter_api_url}/swap"
        try:
            # Get user's public key from private key
            signer = Keypair.from_secret_key(base64.b64decode(self.private_key))
            user_public_key = str(signer.public_key)

            # Prepare the swap request body
            swap_request = {
                "quoteResponse": quote_response,
                "userPublicKey": user_public_key,
                "wrapUnwrapSOL": True  # Automatically wrap/unwrap SOL
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=swap_request) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        self.logger.warning(f"Failed to get swap transaction, status code: {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error getting swap transaction: {e}")
            return None

    async def sign_transaction(self, transaction_data):
        """Sign a Solana transaction using the bot's private key."""
        try:
            # Create transaction from serialized data
            transaction = Transaction.deserialize(transaction_data['transaction'])
            
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
            response = await self.client.send_transaction(
                transaction,
                opts=TxOpts(
                    skip_preflight=True,  # Skip preflight to avoid false negatives
                    max_retries=3
                )
            )
            return response
        except Exception as e:
            self.logger.error(f"Error sending transaction: {e}")
            return None

    async def confirm_transaction(self, signature):
        """Confirm transaction via Solana websocket."""
        try:
            # Subscribe to transaction confirmation
            async with self.client.ws_subscribe('signatureSubscribe', [signature, {"commitment": "confirmed"}]) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if 'result' in data and data['result'].get('value', {}).get('err') is None:
                            self.logger.info(f"Transaction {signature} confirmed")
                            return True
                        elif 'result' in data and data['result'].get('value', {}).get('err'):
                            self.logger.error(f"Transaction {signature} failed: {data['result']['value']['err']}")
                            return False
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        self.logger.error(f"Websocket error: {msg.data}")
                        break
        except Exception as e:
            self.logger.error(f"Error confirming transaction: {e}")
            return False

    async def execute_trade(self, input_token, output_token, amount):
        """Execute a token swap via Jupiter."""
        try:
            # Get swap quote
            quote = await self.get_swap_quote(input_token, output_token, amount)
            if not quote:
                self.logger.error(f"Failed to retrieve quote for {input_token} -> {output_token}")
                return None

            # Get swap transaction
            swap_tx_data = await self.get_swap_transaction(quote)
            if not swap_tx_data:
                self.logger.error("Failed to get swap transaction data")
                return None

            # Sign transaction
            signed_tx = await self.sign_transaction(swap_tx_data)
            if not signed_tx:
                self.logger.error("Failed to sign transaction")
                return None

            # Send transaction
            response = await self.send_transaction(signed_tx)
            if not response or not response.get('result'):
                self.logger.error("Failed to send transaction")
                return None

            # Confirm transaction
            signature = response['result']
            confirmed = await self.confirm_transaction(signature)
            
            if confirmed:
                self.logger.info(f"Swap {input_token} -> {output_token} executed successfully!")
                return {
                    'success': True,
                    'signature': signature,
                    'input_amount': amount,
                    'expected_output_amount': quote['outAmount'],
                    'price_impact_pct': quote['priceImpactPct']
                }
            else:
                self.logger.error(f"Transaction {signature} failed to confirm")
                return None

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None

    async def get_order_status(self, signature):
        """Get the status of a transaction by its signature."""
        try:
            response = await self.client.get_signature_statuses([signature])
            if response and 'result' in response:
                status = response['result']['value'][0]
                if status is None:
                    return "not_found"
                elif status['err']:
                    return "failed"
                elif status['confirmationStatus'] == "finalized":
                    return "confirmed"
                else:
                    return "pending"
            return "error"
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return "error"

    async def close(self):
        """Close the Solana RPC client."""
        await self.client.close()
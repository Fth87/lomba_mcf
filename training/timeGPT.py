from nixtla import NixtlaClient
import os
from dotenv import load_dotenv
load_dotenv()


nixtla_client = NixtlaClient(
    base_url = 'https://api-preview.nixtla.io',  # Needed for TimeGPT-2 family
    api_key=os.getenv('nixtla_key')
)


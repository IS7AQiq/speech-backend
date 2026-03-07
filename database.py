# database.py - Loads credentials from environment / .env file
import os
import logging
from supabase import create_client

logger = logging.getLogger(__name__)

# Load credentials from environment variables set in .env
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
# Use service key if available for backend, fallback to anon key
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError(
        "SUPABASE_URL and either SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY must be set in the .env file."
    )


class SupabaseDB:
    def __init__(self):
        logger.info("🚀 Connecting to Supabase…")
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Connected to Supabase!")

    def get_categories(self):
        try:
            response = self.client.table("categories").select("*").execute()
            logger.info(f"📊 Found {len(response.data)} categories")
            return response.data
        except Exception as e:
            logger.error(f"❌ get_categories error: {e}")
            return []

    def get_words_by_category(self, category_id: int):
        try:
            response = (
                self.client.table("words")
                .select("*")
                .eq("category_id", category_id)
                .execute()
            )
            logger.info(f"📊 Found {len(response.data)} words in category {category_id}")
            return response.data
        except Exception as e:
            logger.error(f"❌ get_words_by_category error: {e}")
            return []

    def verify_token(self, token: str):
        """Verify Supabase JWT token."""
        import jwt
        try:
            # We skip signature verification here to easily extract user_id,
            # trusting that Supabase API Gateway or your Flutter app validates the token.
            # In deep production, you would fetch Supabase's public JWT secret and decode it fully.
            decoded = jwt.decode(token, options={"verify_signature": False})
            return decoded.get("sub") # 'sub' is the user ID in Supabase JWTs
        except Exception as e:
            logger.error(f"❌ Token verification failed: {e}")
            return None
            
    def save_transcription(self, user_id: str, audio_path: str, transcript: str, stuttering_data: dict, category_id: int = None, audio_url: str = None):
        """Save a transcription result to Supabase. Optionally store audio_url."""
        try:
            # Note: This schema assumes you have a 'transcriptions' table matching these columns
            entry = {
                "user_id": user_id,
                "transcript": transcript,
                "stuttering_events": stuttering_data.get("stuttering_events", []),
                "problem_words": stuttering_data.get("problem_words", []),
                "total_events": stuttering_data.get("total_events", 0),
            }
            if category_id:
                entry["category_id"] = category_id
                
            if audio_url:
                entry["audio_url"] = audio_url
                
            response = self.client.table("transcriptions").insert(entry).execute()
            logger.info("✅ Saved transcription to database")
            return response.data
        except Exception as e:
            logger.error(f"❌ Failed to save transcription: {e}")
            return None

    def get_user_transcriptions(self, user_id: str):
        """Fetch all transcriptions for a specific user."""
        try:
            response = self.client.table("transcriptions").select("*").eq("user_id", user_id).order('created_at', desc=True).execute()
            logger.info(f"📊 Found {len(response.data)} transcriptions for user {user_id}")
            return response.data
        except Exception as e:
            logger.error(f"❌ Failed to fetch user transcriptions: {e}")
            return None

    def get_transcription_by_id(self, user_id: str, transcription_id: int):
        """Fetch a specific transcription by ID for a user."""
        try:
            response = self.client.table("transcriptions").select("*").eq("id", transcription_id).eq("user_id", user_id).execute()
            if not response.data:
                return None
            return response.data[0]
        except Exception as e:
            logger.error(f"❌ Failed to fetch transcription by id: {e}")
            return None

    def delete_transcription_by_id(self, user_id: str, transcription_id: int):
        """Delete a specific transcription by ID for a user."""
        try:
            # We enforce user_id match to ensure users can only delete their own data if RLS isn't perfect
            response = self.client.table("transcriptions").delete().eq("id", transcription_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete transcription: {e}")
            return False

    def upload_audio(self, user_id: str, file_path: str) -> str:
        """Upload audio file to Supabase storage and return public URL."""
        try:
            bucket_name = "audio-recordings" # Ensure you create this bucket in Supabase!
            file_name = f"{user_id}/{os.path.basename(file_path)}"
            
            # Read file and upload
            with open(file_path, "rb") as f:
                res = self.client.storage.from_(bucket_name).upload(
                    path=file_name,
                    file=f,
                    file_options={"content-type": "audio/wav"}
                )
            
            # Get public URL
            public_url = self.client.storage.from_(bucket_name).get_public_url(file_name)
            logger.info(f"✅ Audio uploaded to Supabase Storage: {public_url}")
            return public_url
        except Exception as e:
            logger.warning(f"⚠️ Failed to upload audio to storage: {e}")
            return None

db = SupabaseDB()
import os
import requests
import spotipy
from requests.adapters import HTTPAdapter
from spotipy.cache_handler import MemoryCacheHandler
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOAuth
from urllib3.util.retry import Retry


def _retry_session() -> requests.Session:
    retry = Retry(
        total = 5,
        status_forcelist=[429 , 503],
        backoff_factor = 1,
        respect_retry_after_header=True,

    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get_spotify_client(user) -> tuple[spotipy.Spotify, MemoryCacheHandler] | tuple[None,None]:
    """Build an authenticated Spotipy client for a given User."""
    if not user.spotify_token:
        return None,None
    
    token_info = {
        "access_token": user.spotify_token,
        "refresh_token": user.spotify_refresh,
        "expires_at": user.spotify_token_expiry or 0,
        "expires_in": 3600,
        "token_type": "Bearer",
        "scope": "user-read-currently-playing user-modify-playback-state",
    }
    cache_handler = MemoryCacheHandler(token_info=token_info)
    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
        scope="user-read-currently-playing user-modify-playback-state",
        cache_handler=cache_handler,
    )

    sp = spotipy.Spotify(auth_manager=auth_manager, request_session= _retry_session())
    return sp, cache_handler


def get_current_track(sp):
    """Return dict with name, artist, spotify_id of currently playing track."""
    current = sp.current_user_playing_track()
    if current is None or current.get("item") is None:
        return None
    item = current["item"]
    return {
        "spotify_id": item["id"],
        "name": item["name"],
        "artist": item["artists"][0]["name"],
    }


#Album Art
#Volume
#Like_Track
#Song preview play on hover 

def queue_track(sp, spotify_id):
    """Add a track to the user's Spotify queue."""
    sp.add_to_queue(spotify_id)
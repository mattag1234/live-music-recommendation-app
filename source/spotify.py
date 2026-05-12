import os
import requests
import spotipy
from requests.adapters import HTTPAdapter
from spotipy.cache_handler import MemoryCacheHandler
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from urllib3.util.retry import Retry


SPOTIFY_SCOPES = (
    "user-read-currently-playing "
    "user-modify-playback-state "
    "user-library-modify"
)


"""CLient Credentials to collect artist metadata"""
_cc_client: spotipy.Spotify | None = None

def _get_cc_client() -> spotipy.Spotify:
    global _cc_client
    if _cc_client is None:
        _cc_client = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        ),
        request_session=_retry_session(),
    )
    return _cc_client


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


def get_spotify_client(user) -> tuple[spotipy.Spotify, MemoryCacheHandler] | tuple[None, None]:
    """Build an authenticated Spotipy client for a given User."""
    if not user.spotify_token:
        return None, None
    
    token_info = {
        "access_token": user.spotify_token,
        "refresh_token": user.spotify_refresh,
        "expires_at": user.spotify_token_expiry or 0,
        "expires_in": 3600,
        "token_type": "Bearer",
        "scope": SPOTIFY_SCOPES,
    }
    cache_handler = MemoryCacheHandler(token_info=token_info)
    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
        scope=SPOTIFY_SCOPES,
        cache_handler=cache_handler,
    )

    sp = spotipy.Spotify(auth_manager=auth_manager, requests_session=_retry_session())
    return sp, cache_handler


def save_tokens_if_refreshed(user, cache_handler: MemoryCacheHandler) -> None:
    """Persist refreshed tokens back to the User row if Spotipy renewed them."""
    from models import db

    cached = cache_handler.get_cached_token()
    if cached and cached["access_token"] != user.spotify_token:
        user.spotify_token = cached["access_token"]
        user.spotify_token_expiry = cached.get("expires_at", 0)
        db.session.commit()


def get_current_track(sp) -> dict | None:
    """Return {"spotify_id", "name", "artist"} for the currently playing track.
    401: Unauthorzied
    403: Forbidden
    Returns None if nothing is playing. """
    try:
        current = sp.current_user_playing_track()
    except SpotifyException as exc:
        if exc.http_status in (401, 403):
            return None
        raise

    if current is None or current.get("item") is None:
        return None

    item = current["item"]
    if item.get("type") != "track":
        return None

    return {
        "spotify_id": item["id"],
        "name": item["name"],
        "artist": item["artists"][0]["name"],
    }


def queue_track(sp, spotify_id: str) -> tuple[bool, str | None]:
    """Add a track to the user's Spotify queue.

    Returns (True, None) on success, or (False, error_message) on failure.
    """
    try:
        sp.add_to_queue(f"spotify:track:{spotify_id}")
        return True, None
    except SpotifyException as exc:
        if exc.http_status == 404:
            return False, "Track not found."
        if exc.http_status == 403:
            return False, "Playback requires an active Spotify Premium device."
        return False, f"Spotify error {exc.http_status}: {exc.msg}"
    

def save_track(sp, spotify_id: str) -> tuple[bool, str | None]:
    """Save a track to the users Spotify account."""
    try:
        sp.current_user_saved_tracks_add(tracks=[spotify_id])
        return True, None
    except SpotifyException as exc:
        if exc.http_status == 403:
            return False, "Saving tracks requires the user-library-modify scope."
        if exc.http_status == 404:
            return False, "Track not found."
        return False, f"Spotify error {exc.http_status}: {exc.msg}"
    



"""Three new functions to fetch meta data """
def get_artist_ids(track_id: str) -> list[str]:
    try:
        track = _get_cc_client(),track(track_id)
        return [artist["id"] for artist in track["artists"]]
    except SpotifyException:
        return []
    
def get_artist_genres(artist_id: str) -> list[str]:
    try:
        artist = _get_cc_client().artist(artist_id)
        return artist.get("genres", [])
    except SpotifyClientCredentials:
        return []
    
def get_track_genres(track_id: str) -> list[str]:
    genres: set[str] = set()
    for artist_id in get_artist_ids(track_id):
        genres.update(get_artist_genres(artist_id))
    return list(genres)
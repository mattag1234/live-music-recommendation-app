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


"""Build an authenticated Spotipy client for a given User."""
def get_spotify_client(user) -> tuple[spotipy.Spotify, MemoryCacheHandler] | tuple[None,None]:

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

    sp = spotipy.Spotify(auth_manager=auth_manager, requests_session=_retry_session())
    return sp, cache_handler

"""Persist refreshed tokens back to the User row if Spotipy renewed them."""
def save_tokens_if_refreshed(user, cache_handler: MemoryCacheHandler) -> None:
    from models import db

    cached = cache_handler.get_cached_token()
    if cached and cached["access_token"] != user.spotify_token:
        user.spotify_token = cached["access_token"]
        user.spotify_token_expiry = cached.get("expires_at", 0)
        db.session.commit()


"""Return {"spotify_id", "name", "artist"} for the currently playing track.

Returns None if nothing is playing.
"""
def get_current_track(sp) -> dict | None:

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

"""Add a track to the user's Spotify queue.

Returns (True, None) on success, or (False, error_message) on failure.
"""
def queue_track(sp, spotify_id: str) -> tuple[bool, str | None]:
    try:
        sp.add_to_queue(f"spotify:track:{spotify_id}")
        return True, None
    except SpotifyException as exc:
        if exc.http_status == 404:
            return False, "Track not found."
        if exc.http_status == 403:
            return False, "Playback requires an active Spotify Premium device."
        return False, f"Spotify error {exc.http_status}: {exc.msg}"
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth


def get_spotify_client(user):
    """Build an authenticated Spotipy client for a given User."""
    if not user.spotify_token:
        return None

    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
        scope="user-read-currently-playing user-modify-playback-state",
        cache_handler=spotipy.cache_handler.CacheHandler(),
    )

    token_info = {
        "access_token": user.spotify_token,
        "refresh_token": user.spotify_refresh,
        "scope": "user-read-currently-playing user-modify-playback-state",
    }
    auth_manager.token_info = token_info
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp


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


def queue_track(sp, spotify_id):
    """Add a track to the user's Spotify queue."""
    sp.add_to_queue(spotify_id)
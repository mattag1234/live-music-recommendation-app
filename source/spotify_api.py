import os
import secrets

from flask import Blueprint, jsonify, redirect, request, session, url_for
from flask_login import current_user, login_required
from spotipy.oauth2 import SpotifyOAuth

import spotify as spotify_lib
from models import db

"""To allow all the Spotify related routes to stay in one file"""
bp = Blueprint("spotify_api", __name__, url_prefix="/api/spotify")

"""Spaces need for concactination, definedhere for local scope"""
_SCOPES = (
    "user-read-currently-playing "
    "user-modify-playback-state "
    "user-library-modify"
)

"""Auth hangshake"""
def _auth_manager(**kwargs) -> SpotifyOAuth:
    return SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
        scope=_SCOPES,
        open_browser=False,
        **kwargs,
    )

"""Checks if account it linked"""
@bp.route("/status")
@login_required
def status():
    return jsonify({"linked": current_user.spotify_token is not None}), 200

"""Initiates OAuth"""
@bp.route("/connect")
@login_required
def connect():
    state = secrets.token_urlsafe(16)
    session["spotify_oauth_state"] = state
    auth_url = _auth_manager(show_dialog=True).get_authorize_url(state=state)
    return redirect(auth_url)


"""Exchanges code for tokens, saves to DB"""
@bp.route("/callback")
@login_required
def callback():
    received_state = request.args.get("state")
    expected_state = session.pop("spotify_oauth_state", None)
    if not received_state or received_state != expected_state:
        return jsonify({"error": "Invalid OAuth state"}), 400

    if request.args.get("error"):
        return redirect(url_for("player_page"))

    code = request.args.get("code")
    if not code:
        return jsonify({"error": "Missing authorization code"}), 400

    token_info = _auth_manager().get_access_token(code, check_cache=False)

    current_user.spotify_token = token_info["access_token"]
    current_user.spotify_refresh = token_info["refresh_token"]
    current_user.spotify_token_expiry = token_info["expires_at"]
    db.session.commit()

    return redirect(url_for("player_page"))

"""Saves track to the user's library"""
@bp.route("/save/<spotify_id>", methods=["POST"])
@login_required
def save_track(spotify_id):
    sp, cache = spotify_lib.get_spotify_client(current_user)
    if sp is None:
        return jsonify({"ok": False, "error": "Spotify account not linked"}), 403

    ok, error = spotify_lib.save_track(sp, spotify_id)
    spotify_lib.save_tokens_if_refreshed(current_user, cache)
    return jsonify({"ok": ok, "error": error}), 200 if ok else 400

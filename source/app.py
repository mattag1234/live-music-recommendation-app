from dotenv import load_dotenv
load_dotenv(override=True)

import os
import secrets

from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from models import db, User
from recommender import recommend
import spotify as spotify_lib
from spotify_api import bp as spotify_bp


def _resolve_database_uri() -> str:
    url = os.getenv('DATABASE_URL')
    if not url:
        return 'sqlite:///music.db'
    # Render/Heroku still hand out legacy "postgres://" URIs; SQLAlchemy 2.x
    # only accepts "postgresql://".
    if url.startswith('postgres://'):
        url = 'postgresql://' + url[len('postgres://'):]
    return url


def _resolve_secret_key() -> str:
    key = os.getenv('SECRET_KEY')
    if key:
        return key
    # Generate a per-process key for local dev. In production this means
    # sessions invalidate on every restart — log a warning so it's visible.
    print('[warning] SECRET_KEY not set; generating an ephemeral one. '
          'Sessions will reset on every restart.')
    return secrets.token_hex(32)


app = Flask(__name__)
app.config['SECRET_KEY'] = _resolve_secret_key()
app.config['SQLALCHEMY_DATABASE_URI'] = _resolve_database_uri()

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login_page'

app.register_blueprint(spotify_bp)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    db.create_all()


# ---------- Pages ----------

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('player_page'))
    return redirect(url_for('login_page'))


@app.route('/login')
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('player_page'))
    return render_template('login.html')


@app.route('/signup')
def signup_page():
    if current_user.is_authenticated:
        return redirect(url_for('player_page'))
    return render_template('signup.html')


@app.route('/player')
@login_required
def player_page():
    return render_template('player.html')


# ---------- Auth ----------

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 400

    new_user = User(username=username)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 201


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid username or password'}), 401

    login_user(user)
    return jsonify({'message': 'Logged in successfully'}), 200


@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200


# ---------- Recommendation + Playback API ----------

def _apply_art(recs):
    """Look up each rec on Spotify by name+artist, attach album art + live ID.

    Kaggle's track IDs have drifted, so we can't trust them for playback or
    art. The recommender's name/artist (from the local CSV) is authoritative,
    and we search Spotify for the current ID matching that name+artist.
    """
    for r in recs:
        hit = spotify_lib.search_track(r.get('name'), r.get('artist'))
        if not hit:
            continue
        if hit.get('album_art'):
            r['album_art'] = hit['album_art']
        # spotify_id is what Queue/Like will use; track_id stays as the dataset ID
        r['spotify_id'] = hit['id']


@app.route('/api/recommend', methods=['POST'])
@login_required
def api_recommend():
    data = request.get_json() or {}
    track_id = data.get('track_id')
    k = int(data.get('k') or 8)  # default 8; client can override

    if not track_id:
        return jsonify({'error': 'track_id required'}), 400

    recs = recommend(track_id, k=k)
    if not recs:
        return jsonify({'recommendations': [], 'message': 'Track not in our catalog'}), 200

    _apply_art(recs)
    return jsonify({'recommendations': recs}), 200


@app.route('/api/recommend/current', methods=['POST'])
@login_required
def api_recommend_current():
    sp, cache = spotify_lib.get_spotify_client(current_user)
    if sp is None:
        return jsonify({'error': 'Spotify not linked'}), 403

    current = spotify_lib.get_current_track(sp)
    if current is None:
        spotify_lib.save_tokens_if_refreshed(current_user, cache)
        return jsonify({'current': None, 'recommendations': []}), 200

    recs = recommend(current['spotify_id'], k=8)
    _apply_art(recs)

    spotify_lib.save_tokens_if_refreshed(current_user, cache)

    return jsonify({
        'current': current,
        'recommendations': recs,
        'in_catalog': len(recs) > 0,
    }), 200


@app.route('/api/current', methods=['GET'])
@login_required
def api_current():
    """Lightweight poll endpoint — just the now-playing track, no recommendations."""
    sp, cache = spotify_lib.get_spotify_client(current_user)
    if sp is None:
        return jsonify({'error': 'Spotify not linked'}), 403

    current = spotify_lib.get_current_track(sp)
    spotify_lib.save_tokens_if_refreshed(current_user, cache)
    return jsonify({'current': current}), 200


@app.route('/api/queue', methods=['POST'])
@login_required
def api_queue():
    data = request.get_json() or {}
    track_id = data.get('track_id')
    if not track_id:
        return jsonify({'ok': False, 'error': 'track_id required'}), 400

    sp, cache = spotify_lib.get_spotify_client(current_user)
    if sp is None:
        return jsonify({'ok': False, 'error': 'Spotify not linked'}), 403

    ok, error = spotify_lib.queue_track(sp, track_id)
    spotify_lib.save_tokens_if_refreshed(current_user, cache)
    return jsonify({'ok': ok, 'error': error}), (200 if ok else 400)


@app.route('/api/skip', methods=['POST'])
@login_required
def api_skip():
    sp, cache = spotify_lib.get_spotify_client(current_user)
    if sp is None:
        return jsonify({'ok': False, 'error': 'Spotify not linked'}), 403

    try:
        sp.next_track()
        ok, error = True, None
    except Exception as exc:
        status = getattr(exc, 'http_status', None)
        if status == 403:
            ok, error = False, 'Skipping requires an active Spotify Premium device.'
        elif status == 404:
            ok, error = False, 'No active Spotify device found.'
        else:
            ok, error = False, str(exc)

    spotify_lib.save_tokens_if_refreshed(current_user, cache)
    return jsonify({'ok': ok, 'error': error}), (200 if ok else 400)


if __name__ == '__main__':
    # Local dev only. Production runs gunicorn (see Dockerfile / Procfile).
    port = int(os.getenv('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=True)

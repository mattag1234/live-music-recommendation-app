from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    spotify_token = db.Column(db.Text, nullable=True)
    spotify_refresh = db.Column(db.Text, nullable=True)
    spotify_token_expiry = db.Column(db.Integer, nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Song(db.Model):
    __tablename__ = "songs"

    id = db.Column(db.Integer, primary_key=True)
    spotify_id = db.Column(db.String(64), unique=True, nullable=False)
    name = db.Column(db.String(256), nullable=False)
    artist = db.Column(db.String(256), nullable=False)

    tempo = db.Column(db.Float, nullable=True)
    energy = db.Column(db.Float, nullable=True)
    danceability = db.Column(db.Float, nullable=True)
    valence = db.Column(db.Float, nullable=True)
    acousticness = db.Column(db.Float, nullable=True)
    loudness = db.Column(db.Float, nullable=True)

class LikedSong(db.Model):
    __tablename__ = "liked_songs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    song_id = db.Column(db.Integer, db.ForeignKey('songs.id'), nullable=False)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import Flask, render_template, Response, jsonify, request, redirect, session

try:
    from camera import *
except Exception as e:
    print("Failed to import camera:", e)
    raise

app = Flask(__name__)
app.secret_key = "emotion_music_secret"

SPOTIFY_CLIENT_ID     = "eeca3142d16d4df4aa8578d7d64c7eef"
SPOTIFY_CLIENT_SECRET = "34f56fb45888447ebd753633476d46c1"
SPOTIFY_REDIRECT_URI  = "http://127.0.0.1:5000/callback"
SCOPE = "streaming user-read-email user-read-private user-read-playback-state user-modify-playback-state"

sp_oauth = SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=SCOPE,
    show_dialog=True
)

headings = ("Name", "Album", "Artist")
df1 = music_rec()
df1 = df1.head(15)

def get_valid_token():
    token_info = session.get("token_info")
    if not token_info:
        return None
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info["refresh_token"])
        session["token_info"] = token_info
    return token_info["access_token"]

@app.route('/')
def index():
    return render_template('index.html', headings=headings, data=df1)

@app.route('/login')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session["token_info"] = token_info
    return redirect('/')

@app.route('/spotify_token')
def spotify_token():
    token = get_valid_token()
    return jsonify({"token": token})

@app.route('/search_track')
def search_track():
    token = get_valid_token()
    if not token:
        return jsonify({"uri": None})
    sp = spotipy.Spotify(auth=token)
    song   = request.args.get('song', '')
    artist = request.args.get('artist', '')
    results = sp.search(q=f"track:{song} artist:{artist}", type="track", limit=1)
    items = results["tracks"]["items"]
    if items:
        track = items[0]
        return jsonify({
            "uri":    track["uri"],
            "name":   track["name"],
            "artist": track["artists"][0]["name"],
            "image":  track["album"]["images"][0]["url"] if track["album"]["images"] else None
        })
    return jsonify({"uri": None})

def gen(camera):
    while True:
        global df1
        frame, df1 = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/t')
def gen_table():
    return df1.to_json(orient='records')

@app.route('/emotion')
def get_emotion():
    from camera import show_text, emotion_dict
    emotion    = emotion_dict[show_text[0]]
    top_song   = df1.iloc[0]['Name']   if not df1.empty else ""
    top_artist = df1.iloc[0]['Artist'] if not df1.empty else ""
    return jsonify({"emotion": emotion, "song": top_song, "artist": top_artist})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)

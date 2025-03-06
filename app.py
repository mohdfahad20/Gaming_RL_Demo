from flask import Flask, Response, send_file
import cv2
import time
from stable_baselines3 import PPO
from vizdoom import DoomGame

app = Flask(__name__)

# Load trained models
models = {
    "10k": PPO.load("basic_model/best_model_10000.zip"),
    "100k": PPO.load("basic_model/best_model_100000.zip")
}

# Doom environment setup
class VizDoomEnv:
    def __init__(self, config="configs/basic.cfg"):
        self.game = DoomGame()
        self.game.load_config(config)
        self.game.set_window_visible(False)  # Hide game window
        self.game.init()

    def reset(self):
        self.game.new_episode()

    def step(self, model):
        if self.game.is_episode_finished():
            self.game.new_episode()
        state = self.game.get_state().screen_buffer
        action, _ = model.predict(state)
        self.game.make_action(action)
        return state

env = VizDoomEnv()

# Streaming live RL gameplay
def generate(model_version):
    model = models[model_version]
    env.reset()
    while True:
        frame = env.step(model)
        if frame is None:
            continue
        frame = cv2.resize(frame, (320, 240))
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)  # Adjust FPS

@app.route("/")
def home():
    return "Flask server is running! Use /stream/10k or /stream/100k to see the RL model in action."

@app.route('/stream/<version>')
def stream(version):
    if version in models:
        return Response(generate(version), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid model version", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

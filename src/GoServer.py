from GoRecord import GoRecord
from flask_restful import Resource, Api
from flask import Flask, request
import os

app = Flask(__name__)
api = Api(app)


class Play(Resource):
    def get(self, sessionId):
        go = GoRecord(sessionId=sessionId)
        return {sessionId: go.get_game_sequence()}

    def put(self, sessionId):
        go = GoRecord(sessionId=sessionId)
        response = go.write_new_step_to_sgf_file(request.form['color'],
                                                 int(request.form['row']),
                                                 int(request.form['col']))
        return {sessionId: response}

    def delete(self, sessionId):
        for file in os.listdir("./"):
            if file.endswith('sfg') and file.split(".")[0] == sessionId:
                os.remove("./"+sessionId+".sfg")


class History(Resource):
    def get(self):
        existing_game_session = {}

        for file in os.listdir("./"):
            if file.endswith('sfg'):
                existing_game_session[str(len(existing_game_session))] = file.split(".")[0]
        return {"Session Count": str(len(existing_game_session)),
                "Existing Session": existing_game_session}

    def delete(self):
        for file in os.listdir("./"):
            if file.endswith('sfg'):
                os.remove("./"+file+".sfg")


api.add_resource(Play, '/sessions/<string:sessionId>')
api.add_resource(History, '/all_sessions')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6900)

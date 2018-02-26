import os
import requests


class BotClient(object):

    """Client for cluster bot."""

    def __init__(self, token=None, job_id=None, url='http://master:8080'):
        if token is None:
            token = os.environ.get('CLUSTER_USER_TOKEN', None)
        self.token = token

        if job_id is None:
            job_id = os.environ.get('SLURM_JOB_ID', None)
        self.job_id = job_id

        self.url = url

    def start(self):
        self.send_status(0)

    def finish(self):
        self.send_status(1)

    def send_status(self, status):
        return requests.post('{}/job/{}/status'.format(self.url, self.job_id),
                params = {'token': self.token}, json={'status': status})

    def send_message(self, message):
        return requests.post('{}/job/{}/message'.format(self.url, self.job_id),
                params = {'token': self.token}, json={'message': message})

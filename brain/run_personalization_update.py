"""
Call this to update the personalization. The facebook access_tokens are stored in mongo
"""
import os
from dotenv import load_dotenv, find_dotenv

# load environment from file if exists
load_dotenv(find_dotenv())

from botcycle import persistence, personalization

def main():
    facebook_users = persistence.get_facebook_users()

    for user in facebook_users:
        personalization.update_facebook_profile(user['_id'])

if __name__ == '__main__':
    main()
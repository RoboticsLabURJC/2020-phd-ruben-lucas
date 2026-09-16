#!/usr/bin/env python3
import os
import sys
import argparse
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/presentations'
]

CLIENT_SECRET_FILE = os.path.join(os.path.dirname(__file__), 'client_secret.json')
TOKEN_FILE = os.path.join(os.path.dirname(__file__), 'token.json')

def get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired Google credentials...")
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Failed to refresh credentials: {e}")
                creds = None
        if not creds:
            if not os.path.exists(CLIENT_SECRET_FILE):
                print(f"Error: Client secret file not found at {CLIENT_SECRET_FILE}")
                print("Please make sure you have copied your client_secret_*.json file to that path.")
                sys.exit(1)
            print("No valid credentials found. Starting authentication flow...")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0, open_browser=False)
        # Save the credentials for the next run
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
            print(f"Credentials saved successfully to {TOKEN_FILE}")
    return creds

def get_drive_service(creds):
    return build('drive', 'v3', credentials=creds)

def get_slides_service(creds):
    return build('slides', 'v1', credentials=creds)

def handle_auth(args):
    try:
        get_credentials()
        print("Authentication successful!")
    except Exception as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)

def handle_search(args):
    creds = get_credentials()
    service = get_drive_service(creds)
    
    query = args.query
    if args.folders_only:
        query = f"mimeType = 'application/vnd.google-apps.folder' and {query}" if query else "mimeType = 'application/vnd.google-apps.folder'"
        
    try:
        results = service.files().list(
            q=query,
            pageSize=args.limit,
            fields="nextPageToken, files(id, name, mimeType, parents)"
        ).execute()
        items = results.get('files', [])
        
        if not items:
            print("No files found.")
            return
            
        print(f"Found {len(items)} files:")
        for item in items:
            parents = ",".join(item.get('parents', []))
            print(f"ID: {item['id']} | Name: {item['name']} | Type: {item['mimeType']} | Parents: [{parents}]")
    except HttpError as error:
        print(f"An error occurred: {error}")

def handle_copy(args):
    creds = get_credentials()
    service = get_drive_service(creds)
    
    body = {}
    if args.name:
        body['name'] = args.name
    if args.folder_id:
        body['parents'] = [args.folder_id]
        
    try:
        print(f"Copying file {args.file_id}...")
        copied_file = service.files().copy(
            fileId=args.file_id,
            body=body
        ).execute()
        print("File copied successfully!")
        print(f"New File ID: {copied_file.get('id')}")
        print(f"New File Name: {copied_file.get('name')}")
    except HttpError as error:
        print(f"An error occurred: {error}")

def handle_create_folder(args):
    creds = get_credentials()
    service = get_drive_service(creds)
    
    file_metadata = {
        'name': args.name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if args.parent_id:
        file_metadata['parents'] = [args.parent_id]
        
    try:
        print(f"Creating folder '{args.name}'...")
        file = service.files().create(body=file_metadata, fields='id').execute()
        print(f"Folder created successfully! ID: {file.get('id')}")
    except HttpError as error:
        print(f"An error occurred: {error}")

def handle_move(args):
    creds = get_credentials()
    service = get_drive_service(creds)
    
    try:
        # Retrieve the existing parents to remove them
        file = service.files().get(fileId=args.file_id, fields='parents').execute()
        previous_parents = ",".join(file.get('parents', []))
        
        # Move the file to the new folder
        print(f"Moving file {args.file_id} to folder {args.folder_id}...")
        file = service.files().update(
            fileId=args.file_id,
            addParents=args.folder_id,
            removeParents=previous_parents,
            fields='id, parents'
        ).execute()
        print("File moved successfully!")
    except HttpError as error:
        print(f"An error occurred: {error}")

def handle_replace_text(args):
    creds = get_credentials()
    service = get_slides_service(creds)
    
    # Check if a single replacement or a JSON map is provided
    replacements = {}
    if args.json_map:
        try:
            replacements = json.loads(args.json_map)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON replacement map: {e}")
            sys.exit(1)
    elif args.search and args.replace is not None:
        replacements[args.search] = args.replace
    else:
        print("Error: Must provide either --search and --replace, or --json-map")
        sys.exit(1)
        
    requests = []
    for search_str, replace_str in replacements.items():
        requests.append({
            'replaceAllText': {
                'containsText': {
                    'text': search_str,
                    'matchCase': True
                },
                'replaceText': replace_str
            }
        })
        
    body = {
        'requests': requests
    }
    
    try:
        print(f"Performing {len(requests)} text replacement(s) on presentation {args.presentation_id}...")
        response = service.presentations().batchUpdate(
            presentationId=args.presentation_id,
            body=body
        ).execute()
        
        # Calculate total replacements made
        total_replacements = 0
        for reply in response.get('replies', []):
            replace_all_text_reply = reply.get('replaceAllText', {})
            occurrences_changed = replace_all_text_reply.get('occurrencesChanged', 0)
            total_replacements += occurrences_changed
            
        print(f"Replacements complete! Total occurrences updated: {total_replacements}")
    except HttpError as error:
        print(f"An error occurred: {error}")

def main():
    parser = argparse.ArgumentParser(description="Google Drive & Slides API Client CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")
    
    # Auth subcommand
    subparsers.add_parser('auth', help="Initiate OAuth authentication flow and save tokens")
    
    # Search subcommand
    search_parser = subparsers.add_parser('search', help="Search for files or folders in Drive")
    search_parser.add_argument('query', nargs='?', default="", help="Query string (e.g. name contains 'Tesis')")
    search_parser.add_argument('--folders-only', action='store_true', help="Filter to only folders")
    search_parser.add_argument('--limit', type=int, default=10, help="Max results to return")
    
    # Copy subcommand
    copy_parser = subparsers.add_parser('copy', help="Copy a Google Drive file")
    copy_parser.add_argument('file_id', help="The ID of the file to copy")
    copy_parser.add_argument('--name', help="The new name for the copied file")
    copy_parser.add_argument('--folder-id', help="The destination folder ID")
    
    # Create folder subcommand
    folder_parser = subparsers.add_parser('create-folder', help="Create a folder in Google Drive")
    folder_parser.add_argument('name', help="The name of the new folder")
    folder_parser.add_argument('--parent-id', help="The parent folder ID (optional)")
    
    # Move subcommand
    move_parser = subparsers.add_parser('move', help="Move a file to a new folder")
    move_parser.add_argument('file_id', help="The ID of the file to move")
    move_parser.add_argument('folder_id', help="The ID of the destination folder")
    
    # Replace text subcommand
    replace_parser = subparsers.add_parser('replace', help="Replace text inside a Google Slides presentation")
    replace_parser.add_argument('presentation_id', help="The ID of the Google Slides presentation")
    replace_parser.add_argument('--search', help="The text pattern to find (e.g., '{{DATE}}')")
    replace_parser.add_argument('--replace', help="The replacement text")
    replace_parser.add_argument('--json-map', help="A JSON object mapping search strings to replacement strings (e.g. '{\\\"{{DATE}}\\\":\\\"2026-09-16\\\"}')")
    
    args = parser.parse_args()
    
    if args.command == 'auth':
        handle_auth(args)
    elif args.command == 'search':
        handle_search(args)
    elif args.command == 'copy':
        handle_copy(args)
    elif args.command == 'create-folder':
        handle_create_folder(args)
    elif args.command == 'move':
        handle_move(args)
    elif args.command == 'replace':
        handle_replace_text(args)

if __name__ == '__main__':
    main()

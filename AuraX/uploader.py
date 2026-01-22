# uploader_v3.py
import requests
import json
import os
import argparse
import getpass
from pathlib import Path
from io import BytesIO

# --- Configuration ---
DEFAULT_API_URL = "http://localhost:8009/ceaf"


def login(api_url: str, username: str, password: str) -> str | None:
    """
    Logs into the Aura API and returns an authentication token.
    Returns None if login fails. (This function remains unchanged)
    """
    login_endpoint = f"{api_url}/auth/login"
    login_payload = {
        "username": username,
        "password": password
    }
    headers = {"Content-Type": "application/json"}

    print(f"Attempting to log in as '{username}'...")
    try:
        response = requests.post(login_endpoint, json=login_payload, headers=headers)

        if response.status_code == 200:
            token = response.json().get("access_token")
            print("✅ Login successful!")
            return token
        else:
            print(f"❌ Login failed. Status code: {response.status_code}")
            print(f"   Reason: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ A network error occurred while trying to log in: {e}")
        return None


def publish_agent(api_url: str, auth_token: str, agent_id: str) -> bool:
    """
    Publishes a private agent to the marketplace, making it a public template.
    Returns True on success, False on failure.
    """
    publish_endpoint = f"{api_url}/agents/{agent_id}/publish"
    headers = {"Authorization": f"Bearer {auth_token}"}

    # O endpoint /publish espera um formulário, mesmo que vazio.
    # Vamos enviar um changelog padrão.
    payload = {
        'changelog': 'Versão inicial publicada via uploader script.'
    }

    print(f"   - Attempting to publish agent {agent_id} to the marketplace...")
    try:
        response = requests.post(publish_endpoint, headers=headers, data=payload)

        if response.status_code == 201:  # O endpoint /publish retorna 201 Created
            public_id = response.json().get("public_template_id")
            print(f"✅ Agent published successfully! Public Template ID: {public_id}")
            return True
        else:
            print(f"❌ Failed to publish agent {agent_id}. Status code: {response.status_code}")
            print(f"   Reason: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ A network error occurred during publishing: {e}")
        return False


def upload_agent_from_biography(api_url: str, auth_token: str, file_path: Path,
                                publish_after_creation: bool = False) -> bool:
    """
    Reads a biography JSON file, uploads it to create a new agent, and optionally publishes it.
    Returns True on overall success, False on failure.
    """
    upload_endpoint = f"{api_url}/agents/from-biography"
    headers = {"Authorization": f"Bearer {auth_token}"}

    print(f"\nProcessing file for creation: {file_path.name}")

    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()

        files_payload = {'file': (file_path.name, BytesIO(file_bytes), 'application/json')}

        print("   - Step 1: Creating new private agent...")
        response = requests.post(upload_endpoint, headers=headers, files=files_payload)

        if response.status_code != 200:
            print(f"❌ Agent creation failed for {file_path.name}. Status code: {response.status_code}")
            print(f"   Reason: {response.text}")
            return False

        # Se chegou aqui, a criação foi um sucesso.
        agent_id_created = response.json().get("agent_id")
        memories_injected = response.json().get("memories_injected", 0)
        print(f"✅ Step 1 Success! Created private agent (ID: {agent_id_created}) with {memories_injected} memories.")

        # Agora, lida com a publicação se a flag estiver ativa
        if publish_after_creation:
            print("   - Step 2: Publishing agent to marketplace...")
            publish_success = publish_agent(api_url, auth_token, agent_id_created)
            # O sucesso geral da operação agora depende do sucesso da publicação
            return publish_success

        # Se não era para publicar, a criação bem-sucedida já é o sucesso final.
        return True

    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ A network error occurred: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred with {file_path.name}: {e}")
        return False


def add_memories_to_agent(api_url: str, auth_token: str, agent_id: str, file_path: Path) -> bool:
    """
    Reads a JSON file with a 'biography' or 'memories' list and adds those memories to an existing agent in Aura V3.
    Returns True on success, False on failure.
    """
    # V3 CHANGE: The endpoint URL is now '/memories/upload'.
    update_endpoint = f"{api_url}/agents/{agent_id}/memories/upload"
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

    print(f"\nProcessing file to update agent: {agent_id}")
    print(f"   - Source file: {file_path.name}")

    try:
        # 1. Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # V3 CHANGE: The payload expects a top-level key "memories".
        # We will gracefully handle if the old key "biography" is still used in the file.
        memories_list = data.get('memories')
        if memories_list is None:
            memories_list = data.get('biography') # Fallback for old format

        if memories_list is None or not isinstance(memories_list, list):
            print(f"❌ Error: The file '{file_path.name}' must contain a top-level key 'memories' or 'biography' with a list of memory objects.")
            return False

        # 3. Prepare the payload with the correct key.
        payload = {
            "memories": memories_list
        }

        # 4. Make the request
        print(f"   - Adding {len(payload['memories'])} new memories to agent...")
        response = requests.post(update_endpoint, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            successful = result.get('successful_uploads', 0)
            failed = result.get('failed_uploads', 0)
            print(f"✅ Success! Upload results for agent {agent_id}: {successful} successful, {failed} failed.")
            if result.get('errors'):
                print(f"   - Details: {result['errors']}")
            return True
        else:
            print(f"❌ Update failed for agent {agent_id}. Status code: {response.status_code}")
            print(f"   Reason: {response.text}")
            return False

    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return False
    except json.JSONDecodeError:
        print(f"❌ Error: Could not parse JSON from {file_path.name}. Please check its format.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ A network error occurred during update: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred with {file_path.name}: {e}")
        return False


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Bulk upload or update agent memories on the Aura V3 platform."
    )
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--create",
        dest="directory",
        type=str,
        help="Path to the directory with .json files to CREATE new agents."
    )
    action_group.add_argument(
        "--add-memories-to",
        dest="agent_id",
        type=str,
        help="The ID of an EXISTING agent to add memories to."
    )

    parser.add_argument(
        "--publish",
        action="store_true",  # Isso torna o argumento uma flag (sem valor)
        help="If used with --create, automatically publish the created agents to the marketplace."
    )

    parser.add_argument(
        "--file",
        type=str,
        help="The single .json file to use for the --add-memories-to action."
    )
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Your Aura AI username."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"The base URL of the Aura AI API (default: {DEFAULT_API_URL})."
    )
    args = parser.parse_args()

    if args.agent_id and not args.file:
        parser.error("--add-memories-to requires --file.")

    password = getpass.getpass(f"Enter password for user '{args.username}': ")

    token = login(args.url, args.username, password)
    if not token:
        return

    if args.directory:
        source_directory = Path(args.directory)
        if not source_directory.is_dir():
            print(f"❌ Error: The specified path '{args.directory}' is not a valid directory.")
            return

        json_files = list(source_directory.glob("*.json"))
        if not json_files:
            print(f"ℹ️ No .json files found in '{source_directory}'. Nothing to do.")
            return

        print(f"\nFound {len(json_files)} JSON file(s) to process for CREATION.")
        success_count = 0
        failure_count = 0
        for file_path in json_files:
            # Passe o valor da flag --publish para a função
            if upload_agent_from_biography(args.url, token, file_path, publish_after_creation=args.publish):
                success_count += 1
            else:
                failure_count += 1

        print("\n" + "=" * 30)
        print("      UPLOAD SUMMARY (CREATE)")
        print("=" * 30)
        print(f"Successful uploads: {success_count}")
        print(f"Failed uploads:     {failure_count}")
        print("=" * 30)

    elif args.agent_id:
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"❌ Error: The specified file '{args.file}' does not exist.")
            return

        add_memories_to_agent(args.url, token, args.agent_id, file_path)


if __name__ == "__main__":
    main()
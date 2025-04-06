import os
import dotenv
from supabase import create_client, Client


class SupabaseClient:
    def __init__(self):
        dotenv.load_dotenv()
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        # Removed the undefined options parameter
        self.client: Client = create_client(self.url, self.key)

    def get_user(self):
        user = self.client.auth.get_user()
        return user

    def sign_up(self, email, password, is_admin=False):
        """
        Sign up a new user with optional admin role

        Args:
            email (str): User's email
            password (str): User's password
            is_admin (bool): Whether the user should have admin role

        Returns:
            User object
        """
        # First create the user
        user_response = self.client.auth.sign_up({"email": email, "password": password})

        # If user creation is successful and admin role is requested
        if user_response:
            # Get the user ID from the response
            user_id = user_response.user.id

            # Update the user's metadata to include the admin role
            # This assumes you have a 'user_roles' table in your Supabase database
            self.client.table('user_roles').insert({
                'user_id': user_id,
                'is_admin': is_admin
            }).execute()

        return user_response

    def sign_in(self, email, password):
        """
        Sign in an existing user

        Args:
            email (str): User's email
            password (str): User's password

        Returns:
            User object
        """
        user = self.client.auth.sign_in_with_password({"email": email, "password": password})
        return user

    def sign_out(self):
        """
        Sign out the current user

        Returns:
            Response object
        """
        response = self.client.auth.sign_out()
        return response

    def get_user_role(self, user_id):
        """
        Get the role of a user

        Args:
            user_id (str): The user ID

        Returns:
            str: The user's role ('admin' or 'user')
        """
        response = self.client.table('user_roles').select('is_admin').eq('user_id', user_id).execute()

        # If there's a role entry and it's 'admin', return 'admin'
        if response.data and len(response.data) > 0 and response.data[0]['is_admin'] == True:
            return 'admin'

        # Default role is 'user'
        return 'user'


# Test main function
def main():
    # Create a Supabase client instance
    supabase_client = SupabaseClient()

    while True:
        print("\nSupabase Authentication Test")
        print("1. Sign Up (Regular User)")
        print("2. Sign Up (Admin User)")
        print("3. Sign In")
        print("4. Get Current User")
        print("5. Get User Role")
        print("6. Sign Out")
        print("7. Exit")

        choice = input("Enter your choice (1-7): ")

        if choice == '1':
            email = input("Enter email: ")
            password = input("Enter password: ")
            try:
                user = supabase_client.sign_up(email, password, is_admin=False)
                print(f"User created: {user}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '2':
            email = input("Enter email: ")
            password = input("Enter password: ")
            try:
                user = supabase_client.sign_up(email, password, is_admin=True)
                print(f"Admin user created: {user}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '3':
            email = input("Enter email: ")
            password = input("Enter password: ")
            try:
                user = supabase_client.sign_in(email, password)
                print(f"User signed in: {user}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '4':
            try:
                user = supabase_client.get_user()
                print(f"Current user: {user}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '5':
            try:
                user = supabase_client.get_user()
                if user:
                    role = supabase_client.get_user_role(user.user.id)
                    print(f"User role: {role}")
                else:
                    print("No user is currently signed in")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '6':
            try:
                response = supabase_client.sign_out()
                print("User signed out")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '7':
            print("Exiting program...")
            break

        else:
            print("Invalid choice, please try again")


if __name__ == "__main__":
    main()
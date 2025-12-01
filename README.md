
# Content Moderation System

A Django-based content moderation system with JWT authentication that checks uploaded documents against company policy documents using AI.

## Features

- **User Authentication**: JWT-based authentication with registration, login, and logout 
- **Policy Management**: Upload and manage policy PDF documents
- **Content Moderation**: Check documents against policies using RAG Pipeline
- **Final Verdict System**: Human-in-the-loop review with approve/reject decisions
- **File Viewing**: View original PDFs before making final decisions
- **Moderation History**: Track all moderation results per user
- **Vector Database**: Uses Chroma for efficient policy document retrieval
- **RESTful API**: Built with Django REST Framework
- **Modern Frontend**: React-based UI with gradients and stuff

## Tech Stack

- **Backend**: Django 4.2, Django REST Framework
- **Frontend**: React, duh
- **Authentication**: JWT (djangorestframework-simplejwt)
- **AI/ML**: LangChain, Groq LLM, Sentence Transformers
- **Vector Database**: ChromaDB
- **PDF Processing**: PyPDF

## Prerequisites

- Python 3.8+
- pip
- Node.js & npm
- Groq API Key

## Installation & Setup

### Backend Setup

#### 1. Clone the repository
```bash
git clone 
cd content-moderation-system
```

#### 2. Create virtual environment
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
# Go grab a coffee, this might take a while
```

#### 4. Set up environment variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your credentials
# - SECRET_KEY: Django secret key
# - GROQ_API_KEY: Your Groq API key
```

#### 5. Run migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

#### 6. Create superuser (highly recommended)
```bash
python manage.py createsuperuser
# You'll need this to access the admin panel
```

#### 7. Run the development server
```bash
python manage.py runserver
# Backend is now alive at http://127.0.0.1:8000/
```

### Frontend Setup

#### 1. Navigate to frontend directory
```bash
cd frontend
```

#### 2. Install dependencies
```bash
npm install
```

#### 3. Create environment file
```bash
# Create frontend/.env
echo "REACT_APP_API_URL=http://127.0.0.1:8000" > .env
```

#### 4. Start the development server
```bash
npm start
# Frontend is now alive at http://localhost:3000/
```

## API Endpoints

### Authentication

#### Register User
```http
POST /api/auth/register/
Content-Type: application/json

{
  "username": "testuser",
  "email": "test@example.com",
  "password": "securepassword123",
  "password2": "securepassword123",
  "first_name": "Test",
  "last_name": "User"
}
```

#### Login
```http
POST /api/auth/login/
Content-Type: application/json

{
  "username": "testuser",
  "password": "securepassword123"
}

Response:
{
  "user": {...},
  "tokens": {
    "refresh": "...",
    "access": "..."
  }
}
```

#### Refresh Token
```http
POST /api/auth/refresh/
Content-Type: application/json

{
  "refresh": "your-refresh-token"
}
```

#### Logout
```http
POST /api/auth/logout/
Authorization: Bearer 
Content-Type: application/json

{
  "refresh_token": "your-refresh-token"
}
```

#### Get User Profile
```http
GET /api/auth/profile/
Authorization: Bearer 
```

### Policy Management

#### Upload Policy Documents
```http
POST /api/moderation/upload-policy/
Authorization: Bearer 
Content-Type: multipart/form-data

files: [policy1.pdf, policy2.pdf]
```

#### List Policies
```http
GET /api/moderation/policies/
Authorization: Bearer 
```

#### Delete Policy
```http
DELETE /api/moderation/policies//
Authorization: Bearer 
```

#### Clear All Policies
```http
POST /api/moderation/clear-policies/
Authorization: Bearer 
```

### Moderation

#### Moderate a File
```http
POST /api/moderation/moderate/
Authorization: Bearer 
Content-Type: multipart/form-data

file: document.pdf
```

#### Get Moderation History
```http
GET /api/moderation/history/
Authorization: Bearer 
```

#### Get Moderation Detail
```http
GET /api/moderation/history//
Authorization: Bearer 
```

#### Update Final Verdict (NEW!)
```http
POST /api/moderation/history//verdict/
Authorization: Bearer 
Content-Type: application/json

{
  "final_verdict": "approved"  // or "rejected"
}
```

## Usage Flow

1. **Register/Login**: Create an account or login to get JWT tokens
2. **Upload Policies**: Upload one or more policy PDF documents
3. **Moderate Content**: Upload a document to check against policies
4. **Review Results**: Check AI-generated moderation results
5. **View Original File**: Click to view the actual PDF 
6. **Make Final Decision**: Approve as Clean or Reject as Violation 
7. **Track History**: View all past moderations and decisions

## Example with cURL

### Register
```bash
curl -X POST http://127.0.0.1:8000/api/auth/register/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpass123",
    "password2": "testpass123"
  }'
```

### Login
```bash
curl -X POST http://127.0.0.1:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123"
  }'
```

### Upload Policy (save the access token from login)
```bash
curl -X POST http://127.0.0.1:8000/api/moderation/upload-policy/ \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "files=@policy1.pdf" \
  -F "files=@policy2.pdf"
```

### Moderate File
```bash
curl -X POST http://127.0.0.1:8000/api/moderation/moderate/ \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "file=@document.pdf"
```

### Make Final Decision
```bash
curl -X POST http://127.0.0.1:8000/api/moderation/history/1/verdict/ \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"final_verdict": "approved"}'
```

## Project Structure

```
content-moderation-system/
├── backend/                  # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── accounts/                 # Authentication app
│   ├── models.py
│   ├── serializers.py
│   ├── views.py
│   └── urls.py
├── moderation/              # Moderation app
│   ├── modules/
│   │   ├── policy_store.py  # Chroma vector store management
│   │   ├── llm.py           # LLM and retrieval chain
│   │   └── moderation_engine.py  # Core moderation logic
│   ├── models.py
│   ├── serializers.py
│   ├── views.py
│   ├── urls.py
│   └── admin.py
├── media/                   # Uploaded files
│   ├── policies/
│   └── moderation_files/
├── policy_store/            # Chroma vector database
├── frontend/                # React based frontend
│   ├── src/
│   │   ├── components/      # Reusable components
│   │   ├── pages/           # Page components
│   │   ├── services/        # API calls
│   │   ├── context/         # React context
│   │   └── utils/           # Helper functions
│   ├── public/
│   └── package.json
├── manage.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md (you are here)
```

## Database Models

### User Profile
- Extends Django's built-in User model
- Tracks creation and update timestamps
- Nothing fancy, just the basics

### Policy Document
- Stores uploaded policy PDFs
- Links to user who uploaded
- Tracks file metadata (size, name, etc.)

### Moderation Result
- Stores results of file moderation
- Links to user who initiated moderation
- Contains verdict and chunk statistics
- **NEW**: Includes `final_verdict` (pending/approved/rejected)
- **NEW**: Includes `reviewed_at` timestamp

### Violation Detail
- Stores specific violations found in chunks
- Links to moderation result
- Contains explanation and policy sources
- Supports VIOLATION, REVIEW, and ERROR verdicts

## Key Features Explained

### RAG Pipeline (Retrieval-Augmented Generation)
The system uses a sophisticated RAG pipeline:
1. **Policy Upload**: PDFs are chunked and embedded into a vector database
2. **Similarity Search**: When moderating, relevant policy chunks are retrieved
3. **LLM Analysis**: Groq LLM analyzes content against retrieved policies
4. **Verdict Generation**: AI generates VIOLATION, REVIEW, or OK verdicts

### Human-in-the-Loop Review
Because AI isn't perfect (yet):
- AI provides initial moderation verdict
- Flags unclear cases as "REVIEW" 
- Human reviews violations and file content
- User makes final approve/reject decision
- System tracks both AI verdict and human decision

### Stats Dashboard
Get the big picture at a glance:
- 🟢 **Clean Files**: Files with no issues
- 🔴 **Violations Found**: Files flagged by AI
- 🟡 **Needs Review**: Unclear cases requiring attention
- 🔵 **Active Policies**: Number of policy documents loaded

## Admin Panel

Access the Django admin panel at `http://127.0.0.1:8000/admin/`

You can:
- View all users and their profiles
- Manage policy documents
- Review moderation results and violations
- Be an admin (feels powerful, doesn't it?)

## Development

### Running Tests
```bash
python manage.py test
# Pray they all pass
```

### Creating Migrations
```bash
python manage.py makemigrations
python manage.py migrate
# Database evolution in action
```

### Collecting Static Files
```bash
python manage.py collectstatic
# For production deployment
```

### Frontend Development
```bash
cd frontend
npm start
# Hot reload enabled, edit and see changes instantly
```

## Troubleshooting

### Policy Store Issues
If you encounter issues with the policy store:
```bash
# Clear the policy store via API
curl -X POST http://127.0.0.1:8000/api/moderation/clear-policies/ \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"

# Or manually nuke it
rm -rf policy_store/*
```

### Database Issues
```bash
# The nuclear option (deletes all data)
rm db.sqlite3
python manage.py migrate
python manage.py createsuperuser
```

### "Module not found" Errors
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend
rm -rf node_modules package-lock.json
npm install
```

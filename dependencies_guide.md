# Dependencies and Setup Guide

## 🎯 Overview

This guide provides complete instructions for setting up all dependencies required to run the Historical Bead Exchange Analysis with Co-DETECT methodology.

## 🖥️ Environment Options

### Option 1: Google Colab (Recommended)
**Advantages**: Pre-configured environment, free GPU/TPU, Google Drive integration
**Setup time**: 5-10 minutes

### Option 2: Local Jupyter Environment  
**Advantages**: Full control, no session limits, custom configurations
**Setup time**: 30-60 minutes

### Option 3: Cloud Platforms (Advanced)
**Options**: AWS SageMaker, Azure ML, Google Cloud AI Platform
**Setup time**: 1-2 hours

---

## 🚀 Option 1: Google Colab Setup (Recommended)

### Prerequisites
- Google account
- Anthropic API key ([Sign up here](https://console.anthropic.com))
- Google Drive with at least 1GB free space

### Step 1: Install Dependencies in Colab
```python
# Copy-paste this into your first Colab cell:

# Install required packages
!pip install anthropic==0.55.0
!pip install scikit-learn==1.4.0
!pip install plotly==5.18.0
!pip install umap-learn==0.5.5
!pip install dash==2.16.1
!pip install jupyter-dash==0.4.2

# Install additional utilities
!pip install openpyxl==3.1.2  # For Excel file handling
!pip install python-dotenv==1.0.0  # For environment variables
!pip install tqdm==4.66.1  # For progress bars

# Verify installations
import anthropic
import sklearn
import plotly
import umap
import pandas as pd
import numpy as np

print("✅ All packages installed successfully!")
print(f"Anthropic version: {anthropic.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Plotly version: {plotly.__version__}")
```

### Step 2: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Create working directory
import os
DRIVE_FOLDER = '/content/drive/MyDrive/CoDetectBeadAnalysis'
os.makedirs(DRIVE_FOLDER, exist_ok=True)
print(f"Working directory created: {DRIVE_FOLDER}")
```

### Step 3: Upload Your Data
```python
# Option A: Upload directly to Colab
from google.colab import files
uploaded = files.upload()  # Upload All_entries_beads.xlsx

# Option B: Place file in Google Drive
# Put your All_entries_beads.xlsx in /content/drive/MyDrive/CoDetectBeadAnalysis/
```

### Step 4: Set API Key
```python
# Secure way to set API key in Colab
from google.colab import userdata
import os

# Option A: Use Colab secrets (recommended)
# 1. Go to Colab secrets panel (🔑 icon)
# 2. Add secret named "ANTHROPIC_API_KEY"
# 3. Use it in code:
API_KEY = userdata.get('ANTHROPIC_API_KEY')

# Option B: Set directly (less secure)
API_KEY = "your_anthropic_api_key_here"

# Verify API key works
import anthropic
client = anthropic.Anthropic(api_key=API_KEY)
print("✅ API key configured successfully!")
```

---

## 💻 Option 2: Local Jupyter Setup

### Prerequisites
- Python 3.8+ ([Download here](https://python.org))
- Git ([Download here](https://git-scm.com))
- Jupyter Notebook or JupyterLab

### Step 1: Create Virtual Environment
```bash
# Create and activate virtual environment
python -m venv bead_analysis_env

# Activate (Windows)
bead_analysis_env\Scripts\activate

# Activate (Mac/Linux)  
source bead_analysis_env/bin/activate
```

### Step 2: Install Dependencies
```bash
# Install core packages
pip install anthropic==0.55.0
pip install scikit-learn==1.4.0
pip install plotly==5.18.0
pip install umap-learn==0.5.5
pip install pandas==2.2.0
pip install numpy==1.26.0
pip install jupyter==1.0.0

# Install additional utilities
pip install openpyxl==3.1.2
pip install python-dotenv==1.0.0
pip install tqdm==4.66.1
pip install matplotlib==3.8.0
pip install seaborn==0.13.0

# Install Jupyter extensions for better visualization
pip install jupyterlab-plotly==0.5.0
```

### Step 3: Create Requirements File
```bash
# Generate requirements.txt
pip freeze > requirements.txt
```

### Step 4: Environment Configuration
```python
# Create .env file for API key
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env

# Load in Python
from dotenv import load_dotenv
import os
load_dotenv()

API_KEY = os.getenv('ANTHROPIC_API_KEY')
```

### Step 5: Start Jupyter
```bash
# Start Jupyter Lab (recommended)
jupyter lab

# Or start classic Jupyter Notebook
jupyter notebook
```

---

## ☁️ Option 3: Cloud Platform Setup

### AWS SageMaker Studio Lab (Free Tier Available)
```python
# In SageMaker terminal:
pip install --user anthropic scikit-learn plotly umap-learn

# Create notebook with kernel: Python 3 (Data Science)
```

### Google Cloud AI Platform
```python
# In AI Platform notebook:
!pip install anthropic scikit-learn plotly umap-learn pandas openpyxl
```

### Azure Machine Learning Studio
```python
# In Azure ML notebook:
%pip install anthropic scikit-learn plotly umap-learn pandas openpyxl
```

---

## 📦 Complete Dependencies List

### Core Dependencies
```txt
anthropic>=0.55.0              # AI annotation and analysis
scikit-learn>=1.4.0            # Machine learning algorithms
plotly>=5.18.0                 # Interactive visualizations
umap-learn>=0.5.5              # Dimensionality reduction
pandas>=2.2.0                  # Data manipulation
numpy>=1.26.0                  # Numerical operations
```

### Data Handling
```txt
openpyxl>=3.1.2                # Excel file support
xlsxwriter>=3.1.9              # Excel writing optimization
python-dotenv>=1.0.0           # Environment variable management
```

### Visualization and UI
```txt
jupyter>=1.0.0                 # Notebook environment
matplotlib>=3.8.0              # Basic plotting
seaborn>=0.13.0                # Statistical visualizations
dash>=2.16.1                   # Interactive web apps (optional)
jupyter-dash>=0.4.2            # Dash in Jupyter (optional)
```

### Utilities
```txt
tqdm>=4.66.1                   # Progress bars
requests>=2.31.0               # HTTP requests
beautifulsoup4>=4.12.2         # HTML parsing (if needed)
python-dateutil>=2.8.2         # Date parsing
```

### Optional (for Advanced Features)
```txt
streamlit>=1.29.0              # Alternative UI framework
gradio>=4.12.0                 # ML model interfaces
huggingface-hub>=0.20.0        # Model hosting (if using HF models)
transformers>=4.36.0           # Alternative NLP models
```

---

## 🔑 API Keys and Authentication

### Required API Keys

#### 1. Anthropic API Key (Required)
```python
# Get from: https://console.anthropic.com
# Pricing: ~$0.025 per text analysis
# Set in environment or Colab secrets

API_KEY = "sk-ant-api03-..."  # Your key here
```

#### 2. OpenAI API Key (Optional - for alternative models)
```python
# Get from: https://platform.openai.com
# Only needed if using GPT models for comparison
OPENAI_API_KEY = "sk-..."  # Optional
```

### Setting API Keys Securely

#### Google Colab (Recommended)
```python
# Use Colab's built-in secrets manager:
# 1. Click the key icon (🔑) in the left sidebar
# 2. Add secret: ANTHROPIC_API_KEY
# 3. Access in code:
from google.colab import userdata
API_KEY = userdata.get('ANTHROPIC_API_KEY')
```

#### Local Development
```python
# Create .env file:
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Load in Python:
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv('ANTHROPIC_API_KEY')
```

#### Environment Variables (Cross-platform)
```bash
# Windows
set ANTHROPIC_API_KEY=your_key_here

# Mac/Linux
export ANTHROPIC_API_KEY=your_key_here
```

---

## 🗂️ Data Requirements

### Input Data Format
Your data should be an Excel file with these columns:

#### Required Columns
```python
required_columns = [
    'text_page_gp',           # Main text content for analysis
    'guid_hash',              # Unique identifier
    'explorer_first_name',    # For context
    'explorer_surname',       # For context
]
```

#### Optional but Recommended
```python
optional_columns = [
    'year_began',             # Time period context
    'year_end',               # Time period context  
    'countries',              # Geographic context
    'title_profession_1',     # Explorer profession
    'highlight',              # Key passages
    'journey_id'              # Journey grouping
]
```

### Data Validation Script
```python
def validate_data_format(file_path):
    """Validate your data meets requirements"""
    
    df = pd.read_excel(file_path)
    
    # Check required columns
    required = ['text_page_gp']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return False
    
    # Check data quality
    text_col = 'text_page_gp'
    valid_texts = df[text_col].notna() & (df[text_col].str.strip() != '')
    
    print(f"✅ Data validation results:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Valid text entries: {valid_texts.sum():,}")
    print(f"   Average text length: {df[text_col].str.len().mean():.0f} characters")
    print(f"   Columns available: {len(df.columns)}")
    
    return True

# Usage:
validate_data_format('All_entries_beads.xlsx')
```

---

## 🚨 Troubleshooting Common Issues

### Installation Problems

#### Issue: Package conflicts
```bash
# Solution: Use fresh environment
pip install --upgrade pip
pip install --force-reinstall anthropic
```

#### Issue: Plotly not displaying in Jupyter
```python
# Solution: Enable Jupyter extensions
import plotly.io as pio
pio.renderers.default = "notebook"  # or "colab" for Colab
```

#### Issue: Memory errors with large datasets
```python
# Solution: Process in smaller batches
BATCH_SIZE = 100  # Reduce from default 800
```

### API Issues

#### Issue: API key not working
```python
# Test API connection:
import anthropic
client = anthropic.Anthropic(api_key="your_key")
try:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": "test"}]
    )
    print("✅ API key working!")
except Exception as e:
    print(f"❌ API error: {e}")
```

#### Issue: Rate limiting
```python
# Solution: Add delays
import time
time.sleep(1)  # Wait 1 second between API calls
```

### Data Issues

#### Issue: Excel file not loading
```python
# Check file format and try alternatives:
try:
    df = pd.read_excel('file.xlsx')
except:
    df = pd.read_csv('file.csv')  # Try CSV instead
```

#### Issue: Text encoding problems
```python
# Solution: Specify encoding
df = pd.read_excel('file.xlsx', encoding='utf-8')
# Or for CSV:
df = pd.read_csv('file.csv', encoding='utf-8', encoding_errors='ignore')
```

---

## 🎛️ Configuration Options

### Performance Tuning
```python
# Adjust these based on your needs:
CONFIG = {
    'SAMPLE_SIZE': 800,                    # Texts per iteration (500-1000 recommended)
    'CONFIDENCE_THRESHOLD': 0.7,           # Edge case cutoff (0.6-0.8 range)
    'MIN_CLUSTER_SIZE': 5,                 # Minimum edge cases per cluster
    'MAX_CLUSTER_SIZE': 20,                # Maximum edge cases per cluster
    'BATCH_SIZE': 10,                      # Save frequency
    'API_DELAY': 0.5,                      # Seconds between API calls
}
```

### Cost Management
```python
# Estimate and control costs:
COST_SETTINGS = {
    'API_COST_PER_REQUEST': 0.025,         # Adjust based on actual costs
    'MAX_BUDGET': 50.0,                    # Maximum spend per iteration
    'COST_ALERTS': True,                   # Warning when approaching budget
}

# Calculate before running:
estimated_cost = SAMPLE_SIZE * API_COST_PER_REQUEST
if estimated_cost > MAX_BUDGET:
    print(f"⚠️ Estimated cost ${estimated_cost:.2f} exceeds budget ${MAX_BUDGET}")
```

### Memory Optimization
```python
# For large datasets:
MEMORY_SETTINGS = {
    'CHUNK_SIZE': 100,                     # Process in chunks
    'CLEAR_CACHE': True,                   # Clear memory between batches
    'REDUCE_PRECISION': True,              # Use float32 instead of float64
}
```

---

## 📋 Complete Requirements File

### requirements.txt
```txt
# Core dependencies
anthropic==0.55.0
scikit-learn==1.4.0
plotly==5.18.0
umap-learn==0.5.5
pandas==2.2.0
numpy==1.26.0

# Data handling
openpyxl==3.1.2
xlsxwriter==3.1.9
python-dotenv==1.0.0

# Visualization
matplotlib==3.8.0
seaborn==0.13.0
jupyter==1.0.0

# Utilities
tqdm==4.66.1
requests==2.31.0
python-dateutil==2.8.2

# Optional interactive features
dash==2.16.1
jupyter-dash==0.4.2
streamlit==1.29.0
gradio==4.12.0

# Text processing utilities
beautifulsoup4==4.12.2
nltk==3.8.1
spacy==3.7.2

# Statistical analysis
scipy==1.12.0
statsmodels==0.14.1
```

### environment.yml (for Conda)
```yaml
name: bead-analysis-codetect
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pandas=2.2.0
  - numpy=1.26.0
  - scikit-learn=1.4.0
  - matplotlib=3.8.0
  - seaborn=0.13.0
  - jupyter=1.0.0
  - pip
  - pip:
    - anthropic==0.55.0
    - plotly==5.18.0
    - umap-learn==0.5.5
    - openpyxl==3.1.2
    - python-dotenv==1.0.0
    - tqdm==4.66.1
    - dash==2.16.1
    - jupyter-dash==0.4.2
```

---

## 🔧 Local Setup Instructions

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/bead-exchange-codetect.git
cd bead-exchange-codetect
```

### Step 2: Environment Setup
```bash
# Option A: Using pip
python -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

pip install -r requirements.txt

# Option B: Using conda
conda env create -f environment.yml
conda activate bead-analysis-codetect
```

### Step 3: Verify Installation
```bash
python -c "
import anthropic, sklearn, plotly, umap, pandas, numpy
print('✅ All core packages imported successfully!')
print(f'Python version: {__import__('sys').version}')
"
```

### Step 4: Start Jupyter
```bash
# Start Jupyter Lab (recommended)
jupyter lab

# Or start classic Jupyter
jupyter notebook
```

---

## 🐳 Docker Setup (Optional)

### Dockerfile
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

### Build and Run
```bash
# Build Docker image
docker build -t bead-analysis-codetect .

# Run container
docker run -p 8888:8888 -v $(pwd)/data:/app/data bead-analysis-codetect
```

---

## 🧪 Testing Your Setup

### Quick Test Script
```python
def test_complete_setup():
    """Test all dependencies and functionality"""
    
    print("🧪 Testing Complete Setup...")
    
    # Test 1: Core imports
    try:
        import anthropic
        import sklearn
        import plotly
        import umap
        import pandas as pd
        import numpy as np
        print("✅ Core packages imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: API connection
    try:
        API_KEY = "your_test_key"  # Replace with real key
        client = anthropic.Anthropic(api_key=API_KEY)
        # Don't actually call API in test
        print("✅ API client created")
    except Exception as e:
        print(f"❌ API setup error: {e}")
        return False
    
    # Test 3: Data processing
    try:
        # Create test dataframe
        test_df = pd.DataFrame({
            'text_page_gp': ['Test text about beads'] * 5,
            'explorer_first_name': ['Test'] * 5
        })
        print("✅ Data processing ready")
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        return False
    
    # Test 4: File I/O
    try:
        import os
        test_dir = '/tmp/test_codetect'
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, 'test.json')
        
        with open(test_file, 'w') as f:
            json.dump({'test': 'data'}, f)
        
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        os.remove(test_file)
        print("✅ File I/O working")
    except Exception as e:
        print(f"❌ File I/O error: {e}")
        return False
    
    print("🎉 All tests passed! Setup is complete.")
    return True

# Run the test
test_complete_setup()
```

---

## 💡 Performance Optimization

### Memory Optimization
```python
# For large datasets:
import gc

def optimize_memory():
    """Optimize memory usage"""
    # Clear unused variables
    gc.collect()
    
    # Use efficient data types
    df = df.astype({
        'text_page_gp': 'string',
        'year_began': 'Int64',  # Nullable integer
        'confidence': 'float32'  # Reduced precision
    })
    
    return df
```

### Speed Optimization
```python
# Parallel processing (local only)
from multiprocessing import Pool
import concurrent.futures

def process_texts_parallel(texts, api_key, n_workers=4):
    """Process multiple texts in parallel"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(analyze_text, text, api_key) for text in texts]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results
```

---

## 📊 Resource Requirements

### Minimum Requirements
- **RAM**: 4GB (for 800 text sample)
- **Storage**: 2GB free space
- **Internet**: Stable connection for API calls
- **Time**: 2-4 hours per iteration

### Recommended Requirements  
- **RAM**: 8GB+ (for full dataset processing)
- **Storage**: 5GB+ free space (for multiple iterations and backups)
- **Internet**: High-speed connection
- **GPU**: Not required but helpful for local clustering

### Cost Estimates
- **Small iteration (800 texts)**: $20-25
- **Medium iteration (2000 texts)**: $50-65  
- **Large iteration (5000 texts)**: $125-165
- **Full dataset (27,000 texts)**: $675-875

---

## 🆘 Support and Troubleshooting

### Common Issues and Solutions

#### "Module not found" errors
```bash
# Reinstall specific package
pip install --force-reinstall anthropic
```

#### "API quota exceeded" errors  
```python
# Add retry logic with exponential backoff
import time
import random

def api_call_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
```

#### Memory errors
```python
# Process in smaller chunks
def process_in_chunks(data, chunk_size=100):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        process_chunk(chunk)
        gc.collect()  # Clear memory
```

### Getting Help
1. **Check logs**: All errors are logged with timestamps
2. **Review progress files**: Check what was last processed
3. **Restart clean**: Clear all variables and restart kernel
4. **Update packages**: Ensure you have latest versions

### Contact and Support
- **GitHub Issues**: [Repository issues page]
- **Documentation**: Check `/docs` folder
- **Research Support**: [your_email@institution.edu]

---

## ✅ Setup Verification Checklist

- [ ] Environment created and activated
- [ ] All packages installed from requirements.txt
- [ ] Jupyter notebook running
- [ ] API key configured and tested
- [ ] Google Drive mounted (if using Colab)
- [ ] Data file accessible and validated
- [ ] Test script runs successfully
- [ ] Working directory created
- [ ] All imports work without errors
- [ ] Sample API call succeeds

---

**Last Updated**: January 2025  
**Compatible With**: Python 3.8+, Google Colab, Jupyter Lab/Notebook  
**Tested On**: Google Colab (recommended), Local Mac/Windows/Linux environments
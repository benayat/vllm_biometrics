# vLLM Biometrics - Face Verification System

A comprehensive face verification system using vLLM (Vision Language Models) for biometric analysis. This project implements multiple prompt strategies and persona-based approaches to maximize accuracy in face verification tasks using the LFW (Labeled Faces in the Wild) dataset.

## 🚀 Features

- **Multiple Prompt Strategies**: Test various prompt approaches including systematic analysis, chain-of-thought, and persona-based prompts
- **Persona-Based Analysis**: Specialized AI personas (Forensic Expert, Security Specialist, Biometric Scientist, etc.)
- **Batch Processing**: Efficient async processing with configurable batch sizes
- **Comprehensive Metrics**: Detailed accuracy analysis with false positive/negative tracking
- **Failure Analysis**: Built-in tools to analyze and improve prompt performance
- **Retry Mechanism**: Robust error handling with exponential backoff
- **Modular Design**: Clean, extensible architecture for easy customization

## 📁 Project Structure

```
vllm_biometrics/
├── README.md
├── vllm_serve_gemma.sh          # vLLM server startup script
├── check_prompts.py             # Main prompt testing script
├── load_pairs.py                # Dataset loading and processing
├── analyze_failures.py          # Failure analysis tool
├── constants/
│   └── prompts.py              # All prompt configurations
├── dataloader/
│   ├── load_dataset.py         # Dataset loading utilities
│   └── util.py                 # Image processing utilities
├── vlm_client/
│   └── client.py               # Async HTTP client for vLLM
└── data/                       # Sample images for testing
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Kaggle API credentials (for dataset download)

### Required Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
vllm>=0.3.0
aiohttp>=3.8.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
pillow>=9.0.0
tqdm>=4.64.0
opendatasets>=0.1.20
asyncio-mqtt>=0.11.0
```

### Kaggle Setup

1. Create a Kaggle account and obtain API credentials
2. Place your `kaggle.json` file in `~/.kaggle/`
3. Set proper permissions: `chmod 600 ~/.kaggle/kaggle.json`

## 🚀 Getting Started

### 1. Start the vLLM Server

First, start the vLLM server with the Gemma model:

```bash
./vllm_serve_gemma.sh
```

This will:
- Download the Gemma-2-27B-IT model (if not already cached)
- Start the server on `http://localhost:8000`
- Enable OpenAI-compatible API endpoints

**Server Configuration:**
- Host: `0.0.0.0`
- Port: `8000`
- Model: `google/gemma-2-27b-it`
- GPU Memory Utilization: 90%

### 2. Test Different Prompts

Run the prompt testing script to evaluate different approaches:

```bash
# Test all prompts with 200 samples
python check_prompts.py --num-samples 200

# Test specific prompts
python check_prompts.py --prompts emotionless_ai forensic_expert --num-samples 400

# Custom batch size and output file
python check_prompts.py --batch-size 16 --output my_results.json --num-samples 100
```

**Available Prompt Types:**
- `concise`: Basic prompt for quick testing
- `enhanced`: Detailed feature comparison
- `systematic`: Step-by-step analysis
- `confidence`: High-confidence standards
- `chain_of_thought`: Reasoning walkthrough
- `emotionless_ai`: Cold, mechanical analysis
- `forensic_expert`: Criminal investigation approach
- `security_specialist`: Security clearance verification
- `biometric_scientist`: Scientific methodology
- `medical_examiner`: Medical precision
- `anti_false_positive`: Reduces false positives
- `ultra_conservative`: Extremely strict matching
- `precision_matching`: Exact feature matching
- `discriminative_analysis`: Focus on differences
- `improved_from_analysis`: Based on failure analysis

### 3. Analyze Results

The system automatically generates comprehensive reports:

```bash
# Results are saved to prompt_comparison_results.json
# View summary in terminal output
```

**Sample Output:**
```
PROMPT COMPARISON SUMMARY
================================================================================
Prompt Name                    Accuracy     FP     FN     Time    
--------------------------------------------------------------------------------
improved_from_analysis         92.45%       28     23     180.2s
forensic_expert               91.80%       31     33     175.8s
anti_false_positive           91.25%       35     40     165.4s
emotionless_ai                90.95%       42     39     170.1s
```

## 📊 Available Prompts

### Basic Prompts
- **Concise**: Simple yes/no question
- **Enhanced**: Detailed feature comparison
- **Systematic**: Step-by-step analysis
- **Confidence**: High-confidence standards
- **Chain of Thought**: Reasoning walkthrough

### Persona-Based Prompts
- **Emotionless AI**: Cold, mechanical precision
- **Forensic Expert**: Criminal investigation approach
- **Security Specialist**: Security clearance verification
- **Biometric Scientist**: Scientific methodology
- **Medical Examiner**: Medical precision

### Specialized Prompts
- **Anti-False Positive**: Reduces incorrect matches
- **Ultra Conservative**: Extremely strict matching
- **Precision Matching**: Exact feature matching
- **Discriminative Analysis**: Focus on differences

## 🔧 Configuration Options

### Command Line Arguments

```bash
python check_prompts.py --help
```

**Options:**
- `--num-samples`: Number of samples to test (default: 200)
- `--batch-size`: Batch size for processing (default: 8)
- `--prompts`: Specific prompts to test (default: all)
- `--output`: Output file for results (default: prompt_comparison_results.json)
- `--max-retries`: Maximum retries for failed requests (default: 3)
- `--timeout`: Request timeout in seconds (default: 30)

### Client Configuration

The VLM client supports various configuration options:

```python
client = Client(
    base_url="http://localhost:8000/v1",
    model_name="gemma-3-27b-it",
    max_retries=3,
    retry_delay=1.0,
    timeout=30
)
```

## 📈 Performance Optimization

### Batch Size Tuning
- **Small batches (4-8)**: More stable, better error handling
- **Large batches (16-32)**: Faster processing, higher memory usage
- **Optimal**: 8-16 for most setups

### Memory Management
- Monitor GPU memory usage
- Adjust `--gpu-memory-utilization` in server script
- Consider model quantization for lower memory usage

### Network Optimization
- Use local deployment for best performance
- Implement request pooling for high throughput
- Monitor network latency and adjust timeouts

## 🔍 Failure Analysis

The system includes built-in failure analysis tools:

```python
# Analyze failed predictions
python analyze_failures.py
```

This generates:
- Detailed failure patterns
- Common error types
- Improved prompt suggestions
- Performance bottlenecks

## 🚨 Troubleshooting

### Common Issues

1. **Server Connection Errors**
   ```bash
   # Check if server is running
   curl http://localhost:8000/health
   
   # Restart server
   ./vllm_serve_gemma.sh
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size
   python check_prompts.py --batch-size 4
   
   # Monitor GPU memory
   nvidia-smi
   ```

3. **Dataset Loading Errors**
   ```bash
   # Check Kaggle credentials
   ls ~/.kaggle/kaggle.json
   
   # Verify permissions
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Unclosed Session Warnings**
   - The latest version includes proper session management
   - Use context managers for manual client usage

### Performance Issues
- Reduce `--num-samples` for faster testing
- Increase `--batch-size` if you have sufficient GPU memory
- Monitor network latency to the vLLM server

## 📝 Example Usage

```python
from vlm_client.client import Client
from constants.prompts import PERSONA_FORENSIC_EXPERT_PROMPT

async def main():
    async with Client() as client:
        result = await client.is_same_person(
            "path/to/image1.jpg",
            "path/to/image2.jpg",
            prompt=PERSONA_FORENSIC_EXPERT_PROMPT
        )
        print(result)

asyncio.run(main())
```

## 📊 Metrics and Evaluation

The system provides comprehensive metrics:

- **Accuracy**: Overall correct predictions
- **False Positives**: Incorrect "same person" predictions
- **False Negatives**: Incorrect "different person" predictions
- **Processing Time**: Total time for batch processing
- **Failed Requests**: Network/server errors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **LFW Dataset**: University of Massachusetts Amherst
- **vLLM**: UC Berkeley Sky Computing Lab
- **Gemma Models**: Google Research

# MMaDA Attention Visualization System

This package provides comprehensive attention visualization and analysis capabilities for MMaDA diffusion language models. It allows you to capture attention weights during text generation and create detailed visualizations and analysis reports.

## 🚀 Features

- **Attention Hooking**: Automatically hooks into attention mechanisms during generation
- **Real-time Capture**: Captures attention weights at each diffusion step
- **Multi-layer Analysis**: Analyzes attention patterns across all transformer layers
- **Comprehensive Visualizations**: Creates heatmaps, evolution plots, and layer comparisons
- **Token-level Analysis**: Analyzes how individual tokens influence generation
- **Statistical Analysis**: Provides detailed statistics and insights about attention patterns

## 📁 Package Structure

```
attention_visualization/
├── __init__.py                 # Package initialization
├── attention_hook.py          # Core attention hooking mechanism
├── attention_visualizer.py    # Visualization generation
├── attention_analyzer.py      # Pattern analysis and statistics
├── analyze_attention.py       # Command-line analysis script
├── test_attention_hook.py     # Test script for the system
└── README.md                  # This documentation
```

## 🔧 Installation

The attention visualization system is part of the MMaDA project. Ensure you have the required dependencies:

```bash
pip install torch numpy matplotlib seaborn pandas
```

## 🎯 Quick Start

### 1. Basic Usage with MMaDA API

```python
from mmada_inference import generate_text

# Generate text with attention visualization enabled
generated_text, history, tokenizer, history_file, attention_maps = generate_text(
    prompt="What is the capital of France?",
    gen_length=128,
    steps=128,
    block_length=32,
    save_history=True,
    visualize_attention=True  # Enable attention capture
)

print(f"Generated text: {generated_text}")
print(f"Captured attention from {len(attention_maps)} modules")
```

### 2. Command-line Usage

```bash
# Generate text with attention visualization
python mmada_api.py "What is the capital of France?" \
    --gen_length 128 \
    --steps 128 \
    --block_length 32 \
    --save_history \
    --visualize_attention

# Analyze captured attention data
python attention_visualization/analyze_attention.py \
    --input logs/generation_with_attention.json \
    --output attention_analysis \
    --max-layers 16 \
    --prompt-length 50
```

## 📊 Visualization Types

### 1. Attention Heatmaps
- **Per-layer heatmaps**: Show attention weights for specific layers
- **Step-by-step evolution**: Track how attention changes across diffusion steps
- **Token-level detail**: Visualize attention between individual tokens

### 2. Layer Comparisons
- **Multi-layer grid**: Compare attention patterns across multiple layers
- **Step-specific views**: Focus on attention patterns at specific generation steps
- **Prompt vs. generated**: Highlight boundaries between input and output

### 3. Attention Evolution
- **Temporal analysis**: Show how attention patterns evolve during generation
- **Before/after comparison**: Compare attention at start vs. end of generation
- **Change visualization**: Highlight differences in attention patterns

## 🔍 Analysis Features

### 1. Attention Pattern Analysis
- **Attention density**: How concentrated vs. diffuse attention is
- **Attention entropy**: How uniform vs. focused attention patterns are
- **Attention focus**: How much attention is on diagonal vs. off-diagonal

### 2. Token Influence Analysis
- **Prompt token influence**: How much each input token affects generation
- **Generated token influence**: How much each output token is influenced
- **Cross-attention patterns**: How prompt and generated tokens interact

### 3. Statistical Summaries
- **Memory usage estimates**: Track memory consumption of attention data
- **Coverage statistics**: Monitor which layers and steps are captured
- **Performance metrics**: Analyze attention capture efficiency

## 🛠️ Advanced Usage

### 1. Custom Attention Hooking

```python
from attention_visualization.attention_hook import AttentionHook

# Create custom attention hook
attention_hook = AttentionHook(model)

# Register hooks
attention_hook.register_hooks()

# Set generation context
attention_hook.set_generation_context(step=0, block=0)

# Generate text (hooks will automatically capture attention)
# ... your generation code ...

# Get captured attention maps
attention_maps = attention_hook.get_attention_maps()

# Clean up
attention_hook.remove_hooks()
```

### 2. Custom Visualization

```python
from attention_visualization.attention_visualizer import AttentionVisualizer

# Create visualizer
visualizer = AttentionVisualizer(output_dir="custom_viz")

# Create specific visualizations
fig = visualizer.create_attention_heatmap(
    attention_weights=attention_maps['layers.0.attention'][0]['attention_weights'],
    layer_name='layers.0.attention',
    step=0,
    block=0
)

# Save visualization
fig.savefig('custom_heatmap.png', dpi=300, bbox_inches='tight')
```

### 3. Custom Analysis

```python
from attention_visualization.attention_analyzer import AttentionAnalyzer

# Create analyzer
analyzer = AttentionAnalyzer(attention_maps)

# Get specific statistics
stats = analyzer.get_attention_statistics()
patterns = analyzer.analyze_attention_patterns()
token_influence = analyzer.analyze_token_influence(prompt_length=50)

# Export custom report
analyzer.export_analysis_report('custom_report.json')
```

## 📈 Performance Considerations

### 1. Memory Usage
- **Attention maps can be large**: For 512 tokens × 512 tokens × 32 layers × 32 heads
- **Estimated memory**: ~2GB for full attention capture
- **Optimization**: Use `max_tokens` parameter to limit analysis scope

### 2. Generation Speed
- **Minimal overhead**: Hooks only capture data, don't modify computation
- **Optional feature**: Only enabled when `visualize_attention=True`
- **Efficient storage**: Attention maps stored in compressed numpy format

### 3. Storage Recommendations
- **Use SSD storage**: For faster I/O when saving large attention files
- **Consider compression**: For long sequences or many layers
- **Clean up old data**: Remove attention files after analysis

## 🔧 Configuration Options

### 1. Visualization Parameters
- **`max_layers`**: Limit number of layers displayed (default: 16)
- **`max_tokens`**: Limit number of tokens analyzed (default: 100)
- **`figsize`**: Control figure dimensions for plots
- **`output_dir`**: Specify where to save visualizations

### 2. Analysis Parameters
- **`prompt_length`**: Length of prompt for token analysis
- **`max_tokens`**: Maximum tokens to include in analysis
- **`include_visualizations`**: Whether to include raw data in reports

### 3. Hook Configuration
- **Automatic registration**: Hooks are automatically registered and cleaned up
- **Error handling**: Graceful fallback if hooks fail to register
- **Memory management**: Automatic cleanup to prevent memory leaks

## 🐛 Troubleshooting

### 1. Common Issues

**"Could not import attention visualization"**
- Ensure the `attention_visualization` package is in your Python path
- Check that all dependencies are installed

**"Failed to register attention hooks"**
- Verify the model has attention mechanisms
- Check for conflicts with existing hooks
- Ensure sufficient memory for hook registration

**"No attention maps found"**
- Confirm `visualize_attention=True` was passed
- Check that generation completed successfully
- Verify hooks were properly registered

### 2. Debug Mode

Enable verbose logging for detailed debugging:

```bash
python analyze_attention.py --input data.json --output analysis --verbose
```

### 3. Memory Issues

If you encounter memory problems:
- Reduce `max_tokens` parameter
- Limit `max_layers` for visualization
- Use smaller generation parameters
- Process attention data in chunks

## 📚 Examples

### 1. Simple Text Generation with Attention

```python
from mmada_inference import generate_text

# Generate with attention visualization
result = generate_text(
    "Explain quantum computing in simple terms",
    gen_length=256,
    steps=256,
    block_length=64,
    save_history=True,
    visualize_attention=True
)

# Extract results
if len(result) == 5:
    text, history, tokenizer, history_file, attention_maps = result
    print(f"Captured attention from {len(attention_maps)} modules")
else:
    text, history, tokenizer, history_file = result
    print("Attention visualization was not enabled")
```

### 2. Comprehensive Analysis

```bash
# Generate text with attention
python mmada_api.py "What is machine learning?" \
    --gen_length 128 \
    --steps 128 \
    --block_length 32 \
    --save_history \
    --visualize_attention

# Analyze the results
python attention_visualization/analyze_attention.py \
    --input logs/generation_with_attention.json \
    --output ml_attention_analysis \
    --max-layers 8 \
    --prompt-length 30 \
    --verbose
```

### 3. Custom Analysis Script

```python
from attention_visualization.attention_analyzer import AttentionAnalyzer
import json

# Load attention data
with open('attention_data.json', 'r') as f:
    attention_maps = json.load(f)

# Create analyzer
analyzer = AttentionAnalyzer(attention_maps)

# Get specific insights
stats = analyzer.get_attention_statistics()
print(f"Total attention parameters: {stats['total_parameters']:,}")

# Analyze specific layer
layer_summary = analyzer.get_layer_attention_summary('layers.0.attention')
if layer_summary:
    print(f"Layer 0 captured {layer_summary['total_captures']} times")
```

## 🤝 Contributing

To contribute to the attention visualization system:

1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests** for new functionality
4. **Update documentation** for any changes
5. **Submit a pull request**

## 📄 License

This attention visualization system is part of the MMaDA project and follows the same license terms.

## 🆘 Support

For issues and questions:

1. **Check the troubleshooting section** above
2. **Review the test examples** in `test_attention_hook.py`
3. **Enable verbose logging** for detailed error messages
4. **Check memory usage** if experiencing crashes

---

**Happy Attention Analysis! 🎉**

The attention visualization system provides unprecedented insight into how MMaDA processes text and makes generation decisions. Use it to understand model behavior, debug generation issues, and advance your research in diffusion language models.

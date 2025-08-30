#!/bin/bash
# Complete Attention Analysis Workflow
# Usage: ./run_attention_analysis.sh [prompt] [output_directory] [max_tokens]
# Examples:
#   ./run_attention_analysis.sh                                    # Default prompt, default output, 25 tokens
#   ./run_attention_analysis.sh "your prompt"                      # Custom prompt, default output, 25 tokens
#   ./run_attention_analysis.sh "your prompt" "custom/output/path"  # Custom prompt and output, 25 tokens
#   ./run_attention_analysis.sh "your prompt" "output" 50          # Custom prompt, output, and 50 tokens

set -e  # Exit on any error

# Parse arguments
PROMPT="${1:-'what is 5 * 7 - 4 * -5?'}"
OUTPUT_DIR="${2:-'outputs/attention_analysis'}"
MAX_TOKENS="${3:-25}"

echo "🎨 Complete Attention Analysis Workflow"
echo "======================================"
echo "Prompt: $PROMPT"
echo "Output directory: $OUTPUT_DIR"
echo "Max tokens per visualization: $MAX_TOKENS"
echo "Generation parameters: gen_length=512, steps=512, block_length=128"
echo ""

# Step 1: Clean up previous outputs
echo "🧹 Cleaning up previous outputs..."
rm -rf "$OUTPUT_DIR"/* 2>/dev/null || true

# Step 2: Generate text with attention capture
echo "📝 Generating text with attention visualization..."
python mmada_api.py "$PROMPT" --visualize_attention --save_history --enable_cot --gen_length 512 --steps 512 --block_length 128

# Step 3: Create Python script for visualization
echo "🎨 Creating attention visualizations..."
cat > temp_attention_script.py << EOF
#!/usr/bin/env python3
from mmada_inference import generate_text
from attention_visualization.attention_visualizer import AttentionVisualizer
from pathlib import Path

# Generate text with attention capture
result = generate_text(
    "$PROMPT",
    visualize_attention=True,
    save_history=True,
    enable_cot=True,
    gen_length=512,
    steps=512,
    block_length=128
)

if len(result) == 5:
    generated_text, history, tokenizer, history_file, attention_maps = result
    print(f"✅ Generated text: {generated_text[:100]}...")
    print(f"✅ Captured attention from {len(attention_maps)} modules")
    
    # Create organized output directory
    output_dir = Path("$OUTPUT_DIR")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer and create plots
    visualizer = AttentionVisualizer(str(output_dir))
    saved_files = visualizer.create_comprehensive_attention_report(
        attention_maps, 
        max_layers=32,
        max_tokens=$MAX_TOKENS
    )
    
    print(f"✅ Created {len(saved_files)} visualization components")
    print(f"📁 Output directory: {output_dir}")
    
    # List generated files
    print(f"\\n📋 Generated structure:")
    for viz_type, filepath in saved_files.items():
        if 'layer_comparison' in viz_type:
            print(f"   📁 {viz_type}: {filepath}/")
        elif 'attention_evolution' in viz_type:
            print(f"   📈 {viz_type}: {filepath}")
        else:
            print(f"   📄 {viz_type}: {filepath}")
else:
    print("❌ Unexpected result format")
EOF

# Step 4: Run visualization script
python temp_attention_script.py

# Step 5: Clean up temporary script
rm -f temp_attention_script.py

# Step 6: Show final results
echo ""
echo "🎉 Attention Analysis Complete!"
echo "=============================="
echo "📁 Output directory: $OUTPUT_DIR"
echo "📊 Total PNG files: $(find $OUTPUT_DIR -name "*.png" | wc -l)"
echo "📈 Step directories: $(find $OUTPUT_DIR -name "step_*" -type d | wc -l)"
echo ""
echo "🔍 Each step folder contains 32 layer files with all attention heads"
echo "🎨 Color scheme: White (no attention) to Dark Red (high attention)"
echo "⚡ Fast generation: Only every 64th step"
echo "📝 Clean titles: Layer X - Step Y"
echo ""
echo "💡 Usage examples:"
echo "   ./run_attention_analysis.sh                                    # Default prompt, default output, 25 tokens"
echo "   ./run_attention_analysis.sh \"your prompt\"                      # Custom prompt, default output, 25 tokens"
echo "   ./run_attention_analysis.sh \"your prompt\" \"custom/output/path\"  # Custom prompt and output, 25 tokens"
echo "   ./run_attention_analysis.sh \"your prompt\" \"output\" 50          # Custom prompt, output, and 50 tokens"

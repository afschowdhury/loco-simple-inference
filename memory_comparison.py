#!/usr/bin/env python3
"""
Comparison between List-based vs String-based memory approaches
for GPT scene descriptions
"""

import json
import time
from typing import List, Dict
from datetime import datetime

class MemoryComparison:
    """
    Compare different memory storage approaches for scene descriptions
    """
    
    def __init__(self):
        # List-based memory (current implementation)
        self.list_memory: List[Dict] = []
        
        # String-based memory (alternative approach)
        self.string_memory: str = ""
        
        self.performance_stats = {
            'list': {'add_times': [], 'context_times': [], 'memory_sizes': []},
            'string': {'add_times': [], 'context_times': [], 'memory_sizes': []}
        }
    
    def add_to_list_memory(self, description: str, max_size: int = 10):
        """Add description to list-based memory"""
        start_time = time.time()
        
        memory_entry = {
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'scene_id': len(self.list_memory) + 1
        }
        
        self.list_memory.append(memory_entry)
        
        # Manage size
        if len(self.list_memory) > max_size:
            excess = len(self.list_memory) - max_size
            self.list_memory = self.list_memory[excess:]
        
        add_time = time.time() - start_time
        self.performance_stats['list']['add_times'].append(add_time)
        self.performance_stats['list']['memory_sizes'].append(len(json.dumps(self.list_memory)))
    
    def add_to_string_memory(self, description: str, max_scenes: int = 10):
        """Add description to string-based memory"""
        start_time = time.time()
        
        scene_id = len(self.string_memory.split('\n---\n')) if self.string_memory else 1
        
        new_entry = f"Scene {scene_id} ({datetime.now().strftime('%H:%M:%S')}): {description}"
        
        if self.string_memory:
            self.string_memory += f"\n---\n{new_entry}"
        else:
            self.string_memory = new_entry
        
        # Manage size by keeping only last N scenes
        scenes = self.string_memory.split('\n---\n')
        if len(scenes) > max_scenes:
            scenes = scenes[-max_scenes:]
            self.string_memory = '\n---\n'.join(scenes)
        
        add_time = time.time() - start_time
        self.performance_stats['string']['add_times'].append(add_time)
        self.performance_stats['string']['memory_sizes'].append(len(self.string_memory))
    
    def get_list_context(self, context_size: int = 3) -> str:
        """Get context from list memory"""
        start_time = time.time()
        
        if not self.list_memory:
            context = ""
        else:
            recent_descriptions = self.list_memory[-context_size:]
            context_parts = []
            for i, desc_data in enumerate(recent_descriptions, 1):
                context_parts.append(f"Previous scene {i}: {desc_data['description']}")
            context = "Context from recent scenes:\n" + "\n".join(context_parts) + "\n\n"
        
        context_time = time.time() - start_time
        self.performance_stats['list']['context_times'].append(context_time)
        return context
    
    def get_string_context(self, context_size: int = 3) -> str:
        """Get context from string memory"""
        start_time = time.time()
        
        if not self.string_memory:
            context = ""
        else:
            scenes = self.string_memory.split('\n---\n')
            recent_scenes = scenes[-context_size:]
            context = "Context from recent scenes:\n" + "\n".join(recent_scenes) + "\n\n"
        
        context_time = time.time() - start_time
        self.performance_stats['string']['context_times'].append(context_time)
        return context
    
    def analyze_coherence(self):
        """Analyze coherence advantages of each approach"""
        
        print("🧠 COHERENCE ANALYSIS")
        print("=" * 50)
        
        print("\n📋 LIST-BASED MEMORY:")
        print("✅ Advantages:")
        print("   - Structured data with timestamps, IDs, metadata")
        print("   - Easy to query specific scenes or time ranges")
        print("   - Supports rich data types (could add confidence scores, etc.)")
        print("   - JSON serializable for persistence")
        print("   - Precise memory management (exact scene count)")
        print("   - Can easily implement advanced features (search, filtering)")
        
        print("❌ Disadvantages:")
        print("   - More complex for GPT to parse as context")
        print("   - Overhead of data structure conversion")
        print("   - Requires formatting when used as prompt context")
        
        print("\n📝 STRING-BASED MEMORY:")
        print("✅ Advantages:")
        print("   - More natural for GPT to understand as context")
        print("   - Direct narrative flow - better coherence")
        print("   - Less processing overhead for context generation")
        print("   - Natural temporal progression in text")
        print("   - Better for storytelling and narrative descriptions")
        
        print("❌ Disadvantages:")
        print("   - Loss of structured metadata")
        print("   - Harder to implement advanced features")
        print("   - Less precise memory management")
        print("   - Potential for text parsing issues")
        print("   - Limited querying capabilities")
    
    def run_performance_test(self, num_descriptions: int = 50):
        """Run performance comparison test"""
        
        print(f"\n⚡ PERFORMANCE TEST ({num_descriptions} descriptions)")
        print("=" * 50)
        
        # Test descriptions
        descriptions = [
            f"A busy street scene with {i} people walking, cars passing, and urban activity during {'daytime' if i % 2 == 0 else 'evening'}"
            for i in range(1, num_descriptions + 1)
        ]
        
        # Test list-based approach
        print("Testing list-based memory...")
        for desc in descriptions:
            self.add_to_list_memory(desc)
            if len(self.list_memory) % 10 == 0:  # Test context every 10 additions
                context = self.get_list_context()
        
        # Reset for string test
        self.string_memory = ""
        
        # Test string-based approach  
        print("Testing string-based memory...")
        for desc in descriptions:
            self.add_to_string_memory(desc)
            if len(self.string_memory.split('\n---\n')) % 10 == 0:  # Test context every 10 additions
                context = self.get_string_context()
    
    def print_performance_results(self):
        """Print performance comparison results"""
        
        import numpy as np
        
        print("\n📊 PERFORMANCE RESULTS")
        print("=" * 50)
        
        # Addition performance
        list_add_avg = np.mean(self.performance_stats['list']['add_times']) * 1000
        string_add_avg = np.mean(self.performance_stats['string']['add_times']) * 1000
        
        print(f"\n⏱️  Addition Time (ms):")
        print(f"   List-based:   {list_add_avg:.3f}ms")
        print(f"   String-based: {string_add_avg:.3f}ms")
        print(f"   Winner: {'String' if string_add_avg < list_add_avg else 'List'} ({abs(list_add_avg - string_add_avg):.3f}ms faster)")
        
        # Context generation performance
        if self.performance_stats['list']['context_times'] and self.performance_stats['string']['context_times']:
            list_context_avg = np.mean(self.performance_stats['list']['context_times']) * 1000
            string_context_avg = np.mean(self.performance_stats['string']['context_times']) * 1000
            
            print(f"\n🔗 Context Generation Time (ms):")
            print(f"   List-based:   {list_context_avg:.3f}ms")
            print(f"   String-based: {string_context_avg:.3f}ms")
            print(f"   Winner: {'String' if string_context_avg < list_context_avg else 'List'} ({abs(list_context_avg - string_context_avg):.3f}ms faster)")
        
        # Memory size
        list_size_avg = np.mean(self.performance_stats['list']['memory_sizes'])
        string_size_avg = np.mean(self.performance_stats['string']['memory_sizes'])
        
        print(f"\n💾 Memory Size (bytes):")
        print(f"   List-based:   {list_size_avg:.0f} bytes")
        print(f"   String-based: {string_size_avg:.0f} bytes")
        print(f"   Winner: {'String' if string_size_avg < list_size_avg else 'List'} ({abs(list_size_avg - string_size_avg):.0f} bytes smaller)")
    
    def show_memory_examples(self):
        """Show examples of both memory formats"""
        
        print("\n📄 MEMORY FORMAT EXAMPLES")
        print("=" * 50)
        
        # Add a few test descriptions
        test_descriptions = [
            "A bustling city intersection with pedestrians crossing during rush hour",
            "A quiet park scene with children playing on swings under afternoon sun",
            "A modern office building entrance with people in business attire"
        ]
        
        # Reset memories
        self.list_memory = []
        self.string_memory = ""
        
        for desc in test_descriptions:
            self.add_to_list_memory(desc, max_size=5)
            self.add_to_string_memory(desc, max_scenes=5)
        
        print("\n📋 LIST-BASED FORMAT:")
        print(json.dumps(self.list_memory, indent=2))
        
        print(f"\n📋 LIST CONTEXT (for GPT):")
        print(self.get_list_context())
        
        print("\n📝 STRING-BASED FORMAT:")
        print(self.string_memory)
        
        print(f"\n📝 STRING CONTEXT (for GPT):")
        print(self.get_string_context())

def main():
    """Run the memory comparison analysis"""
    
    print("🔍 GPT Scene Description Memory Comparison")
    print("=" * 60)
    
    comparison = MemoryComparison()
    
    # Show format examples
    comparison.show_memory_examples()
    
    # Analyze coherence
    comparison.analyze_coherence()
    
    # Run performance test
    comparison.run_performance_test(30)
    
    # Show results
    comparison.print_performance_results()
    
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    print("🎯 FOR BETTER COHERENCE & NATURAL FLOW:")
    print("   → Use STRING-BASED memory")
    print("   → Better for narrative continuity")
    print("   → More natural for GPT context")
    
    print("\n🔧 FOR ADVANCED FEATURES & FLEXIBILITY:")
    print("   → Use LIST-BASED memory")
    print("   → Better for complex queries")
    print("   → More extensible architecture")
    
    print("\n⚖️  HYBRID APPROACH:")
    print("   → Keep list for functionality")
    print("   → Generate string context for GPT")
    print("   → Best of both worlds!")

if __name__ == "__main__":
    main()

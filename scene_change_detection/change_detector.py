import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import AutoModel, AutoProcessor
import timm
import cv2
import numpy as np
import os

# Set OpenCV to use a headless backend to avoid Qt issues
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from typing import Tuple, Optional, Union
import time
from sklearn.metrics.pairwise import cosine_similarity

class SceneChangeDetector:
    """
    Real-time scene change detection using MobileCLIP-S1 or FasterViT-0
    Optimized for RTX 4070 SUPER with TensorRT-ready implementation
    """
    
    def __init__(self, 
                 model_name: str = "mobileclip",  # "mobileclip" or "fastervit"
                 similarity_threshold: float = 0.85,
                 device: str = "cuda",
                 use_fp16: bool = True,
                 enable_tensorrt: bool = True):
        """
        Initialize the scene change detector
        
        Args:
            model_name: "mobileclip" for MobileCLIP-S1 or "fastervit" for FasterViT-0
            similarity_threshold: Threshold for scene change (lower = more sensitive)
            device: Device to run inference on
            use_fp16: Use FP16 mixed precision for speed
            enable_tensorrt: Enable TensorRT optimizations (requires torch_tensorrt)
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.enable_tensorrt = enable_tensorrt
        
        # Initialize model and processor
        self.model = None
        self.processor = None
        self.previous_embedding = None
        
        # Performance tracking
        self.inference_times = []
        
        # Load the specified model
        self._load_model()
        
        print(f"✅ Scene Change Detector initialized with {model_name}")
        print(f"📱 Device: {self.device}")
        print(f"⚡ FP16: {self.use_fp16}")
        print(f"🚀 TensorRT: {self.enable_tensorrt}")
    
    def _load_model(self):
        """Load and optimize the specified model"""
        try:
            if self.model_name == "mobileclip":
                self._load_mobileclip()
            elif self.model_name == "fastervit":
                self._load_fastervit()
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
            # Apply optimizations
            self._optimize_model()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _load_mobileclip(self):
        """Load MobileCLIP-S1 model"""
        try:
            # Using Hugging Face transformers
            model_id = "apple/MobileCLIP-S1"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            
            # Alternative: Direct implementation if above doesn't work
            # You might need to clone the official repo and load weights manually
            
        except Exception as e:
            print(f"⚠️  Hugging Face model not available, using alternative loading...")
            # Fallback to manual implementation
            self._load_mobileclip_manual()
    
    def _load_mobileclip_manual(self):
        """Manual MobileCLIP loading (fallback)"""
        # This would require downloading weights from the official repo
        # For now, using a compatible vision transformer
        self.model = timm.create_model('mobilevitv2_050', pretrained=True, num_classes=0)
        
        # Create manual processor
        self.processor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_fastervit(self):
        """Load FasterViT-0 model"""
        try:
            # Using timm for FasterViT
            self.model = timm.create_model('fastervit_0_224', pretrained=True, num_classes=0)
            
            # Create processor for FasterViT
            self.processor = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            print(f"⚠️  FasterViT not available in timm, using alternative...")
            # Fallback to efficient alternative
            self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
            self.processor = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _optimize_model(self):
        """Apply performance optimizations"""
        # Move to device
        self.model = self.model.to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Enable FP16 if requested
        if self.use_fp16:
            self.model = self.model.half()
        
        # Compile model for optimization (PyTorch 2.0+)
        try:
            self.model = torch.compile(self.model, mode="max-autotune")
            print("✅ Model compiled with torch.compile")
        except:
            print("⚠️  torch.compile not available, skipping compilation")
        
        # TensorRT optimization (if available)
        if self.enable_tensorrt:
            try:
                import torch_tensorrt
                # This would require proper TensorRT setup
                print("✅ TensorRT optimization available")
            except ImportError:
                print("⚠️  TensorRT not available, skipping TensorRT optimization")
    
    def _extract_features(self, frame: np.ndarray) -> torch.Tensor:
        """Extract features from a frame"""
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess frame
        if hasattr(self.processor, 'preprocess'):
            # For transformers-based processor
            inputs = self.processor(images=frame, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)
        else:
            # For torchvision transforms
            if isinstance(frame, np.ndarray):
                frame = frame.astype(np.uint8)
            
            processed_frame = self.processor(frame)
            pixel_values = processed_frame.unsqueeze(0).to(self.device)
        
        # Apply FP16 if enabled
        if self.use_fp16:
            pixel_values = pixel_values.half()
        
        # Extract features
        with torch.no_grad():
            if self.model_name == "mobileclip" and hasattr(self.model, 'get_image_features'):
                # For official MobileCLIP
                features = self.model.get_image_features(pixel_values)
            else:
                # For timm models
                features = self.model(pixel_values)
        
        # Normalize features for cosine similarity
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def detect_scene_change(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect scene change in the given frame
        
        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)
            
        Returns:
            Tuple of (is_scene_changed: bool, frame_if_changed: Optional[np.ndarray])
        """
        start_time = time.time()
        
        try:
            # Extract features from current frame
            current_embedding = self._extract_features(frame)
            
            # For the first frame, store embedding and return no change
            if self.previous_embedding is None:
                self.previous_embedding = current_embedding
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                return False, None
            
            # Calculate similarity between current and previous frame
            similarity = self._calculate_similarity(current_embedding, self.previous_embedding)
            
            # Determine if scene changed
            scene_changed = similarity < self.similarity_threshold
            
            # Update previous embedding
            self.previous_embedding = current_embedding
            
            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Return result
            if scene_changed:
                return True, frame.copy()
            else:
                return False, None
                
        except Exception as e:
            print(f"❌ Error in scene change detection: {e}")
            return False, None
    
    def _calculate_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Convert to CPU and numpy for sklearn
        emb1_cpu = embedding1.cpu().numpy()
        emb2_cpu = embedding2.cpu().numpy()
        
        # Calculate cosine similarity
        similarity = cosine_similarity(emb1_cpu, emb2_cpu)[0][0]
        
        return float(similarity)
    
    def switch_model(self, new_model_name: str):
        """Switch between MobileCLIP and FasterViT"""
        if new_model_name != self.model_name:
            print(f"🔄 Switching from {self.model_name} to {new_model_name}")
            self.model_name = new_model_name
            self.previous_embedding = None  # Reset previous embedding
            self._load_model()
            print(f"✅ Successfully switched to {new_model_name}")
        else:
            print(f"⚠️  Already using {new_model_name}")
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {"message": "No inference performed yet"}
        
        avg_time = np.mean(self.inference_times[-100:])  # Last 100 frames
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            "model": self.model_name,
            "average_inference_time_ms": avg_time * 1000,
            "fps": fps,
            "device": str(self.device),
            "fp16_enabled": self.use_fp16,
            "total_inferences": len(self.inference_times)
        }
    
    def set_threshold(self, new_threshold: float):
        """Update similarity threshold"""
        self.similarity_threshold = new_threshold
        print(f"🎯 Threshold updated to {new_threshold}")

# Example usage and testing
def test_scene_change_detector(show_video=False):
    """Test the scene change detector with sample video processing
    
    Args:
        show_video: Whether to display video frames (requires display server)
    """
    
    # Initialize detector with MobileCLIP
    detector = SceneChangeDetector(
        model_name="mobileclip",
        similarity_threshold=0.85,
        use_fp16=True
    )
    
    # Simulate video processing
    print("\n🎬 Starting video processing simulation...")
    
    # Use video file from videos folder
    video_path = "/home/cmuser/ASIF/loco-simple/videos/demo_data.mp4"
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    scene_changes = 0
    max_frames = 300  # Limit for testing
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("📹 End of video reached")
                break
            
            # Detect scene change
            is_changed, changed_frame = detector.detect_scene_change(frame)
            
            if is_changed:
                scene_changes += 1
                print(f"🔄 Scene change detected at frame {frame_count}")
                
                # You can save or process the changed frame here
                # cv2.imwrite(f"scene_change_{scene_changes}.jpg", changed_frame)
            
            frame_count += 1
            
            # Progress report every 50 frames
            if frame_count % 50 == 0:
                print(f"🎬 Processed {frame_count} frames, detected {scene_changes} scene changes")
            
            # Display frame (optional) - only if show_video is True
            if show_video:
                try:
                    cv2.imshow('Video', frame)
                    # Break on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"⚠️  Display error: {e}. Continuing without display...")
                    show_video = False  # Disable display for remaining frames
            
            # Test model switching every 100 frames
            if frame_count == 100:
                detector.switch_model("fastervit")
            elif frame_count == 200:
                detector.switch_model("mobileclip")
                
    except KeyboardInterrupt:
        print("\n⏹️  Processing stopped by user")
    
    finally:
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # Print performance statistics
        stats = detector.get_performance_stats()
        print(f"\n📊 Performance Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n📈 Total scene changes detected: {scene_changes}")

if __name__ == "__main__":
    test_scene_change_detector()
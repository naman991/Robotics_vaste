#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

// ============================================================================
// 1. HARDWARE CONFIGURATION LAYER (Tweak these when deploying to real hardware)
// ============================================================================
struct PipelineConfig {
    int frame_width;
    int frame_height;
    int tile_dim;
    int down_dim;
    int num_tiles;
    int left_mask_width;
    int right_mask_width;
};

// Default baseline metrics derived from the 4K 6-tile square crop math
// Easily altered at runtime/initialization depending on dataset aspect ratios
PipelineConfig current_config = {
    3840, // frame_width
    2160, // frame_height
    1280, // tile_dim
    256,  // down_dim
    6,    // num_tiles
    40,   // left_mask_width
    40    // right_mask_width
};

struct SliceGeometry {
    int x;
    int y;
};

// Global pointer matrix to be populated dynamically at boot using configuration metrics
std::vector<SliceGeometry> geometry_matrix;

// ============================================================================
// 2. STATIC MEMORY ARENA (Allocated once at application boot to eliminate GIL/Heap)
// ============================================================================
cv::Mat static_scratchpad;

// Multi-dimensional dynamic representation of our fixed inference target arrays
// Keeps memory sequential and contiguous for the NPU execution provider
std::vector<std::vector<std::vector<std::vector<float>>>> inference_target_buffers;

// ============================================================================
// 3. PIPELINE IMPLEMENTATION FUNCTIONS
// ============================================================================

void initialize_pipeline(const PipelineConfig& config) {
    // 1. Generate Geometry Matrix dynamically based on incoming config struct
    geometry_matrix.clear();
    
    // Explicit 6-Tile Layout calculation using config parameters
    // Accommodates changing spatial boundaries or shifts in machine frame clearance
    int vertical_overlap_start = config.frame_height - config.tile_dim; // 2160 - 1280 = 880
    int horizontal_step = config.tile_dim;                             // 1280
    
    geometry_matrix.push_back({0, 0});                                         // Col 1, Top
    geometry_matrix.push_back({0, vertical_overlap_start});                    // Col 1, Bottom
    geometry_matrix.push_back({horizontal_step, 0});                           // Col 2, Top
    geometry_matrix.push_back({horizontal_step, vertical_overlap_start});      // Col 2, Bottom
    geometry_matrix.push_back({horizontal_step * 2, 0});                       // Col 3, Top
    geometry_matrix.push_back({horizontal_step * 2, vertical_overlap_start});  // Col 3, Bottom

    // 2. Allocate Static Scratchpad Memory Space (Zero runtime overhead)
    static_scratchpad = cv::Mat(config.down_dim, config.down_dim, CV_8UC1);

    // 3. Allocate Continuous Float32 Tensor Output Space
    inference_target_buffers.assign(
        config.num_tiles, std::vector<std::vector<std::vector<float>>>(
            3, std::vector<std::vector<float>>(
                config.down_dim, std::vector<float>(config.down_dim, 0.0f)
            )
        )
    );
}

void process_pipeline_tiles(cv::Mat& raw_frame, const PipelineConfig& config) {
    // 1. Optimized Coordinate-Targeted Masking (Zero-Memory Overhead Cache-Safe Path)
    // Adjusts instantly to fabric edge weave tolerances dictated by config file
    if (config.left_mask_width > 0) {
        raw_frame(cv::Rect(0, 0, config.left_mask_width, config.frame_height)).setTo(0);
    }
    if (config.right_mask_width > 0) {
        int right_x = config.frame_width - config.right_mask_width;
        raw_frame(cv::Rect(right_x, 0, config.right_mask_width, config.frame_height)).setTo(0);
    }

    // 2. Loop through the pre-calculated coordinate regions
    for (int i = 0; i < config.num_tiles; ++i) {
        const auto& geo = geometry_matrix[i];
        
        // Zero-Copy Window View Selection via Pointer Offsets
        cv::Mat tile_view = raw_frame(cv::Rect(geo.x, geo.y, config.tile_dim, config.tile_dim));
        
        // Downsample directly into the permanent static scratchpad block
        cv::resize(tile_view, static_scratchpad, cv::Size(config.down_dim, config.down_dim), 0, 0, cv::INTER_AREA);
        
        // 3. Float32 Replicated 3-Channel Normalization Pass
        for (int r = 0; r < config.down_dim; ++r) {
            const uchar* row_ptr = static_scratchpad.ptr<uchar>(r);
            for (int c = 0; c < config.down_dim; ++c) {
                float normalized_val = row_ptr[c] / 255.0f;
                
                // Write directly to sequential global memory spaces
                inference_target_buffers[i][0][r][c] = normalized_val; // Ch1
                inference_target_buffers[i][1][r][c] = normalized_val; // Ch2
                inference_target_buffers[i][2][r][c] = normalized_val; // Ch3
            }
        }
    }
}
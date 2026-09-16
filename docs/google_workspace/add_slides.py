#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from google_workspace.gapi_client import get_credentials, get_slides_service

PRESENTATION_ID = '1RHfwlYqdIFlw3tLAaJ_JH6rVCRP6D1NyMFhW667EiNY'

SLIDES_DATA = [
    {
        'id': 'slide_perception_state_s2r',
        'title': 'S2R-ACC: Perception & State Representation',
        'body': (
            "• Perception Engine: Hybrid YOLOPv1 + Classical Vision\n"
            "   - Camera mounted at x=1.5, z=1.2 (640x480). Retrained YOLOP model on GPU.\n"
            "   - Filtering: Morphology closing closing kernels & contour filtering for clean lane-line binary masks.\n"
            "   - Spline Fitting: 3rd-degree cubic polynomial (x = Ay³ + By² + Cy + D) with RANSAC outlier rejection; curvature-adaptive fallbacks.\n"
            "   - Bottom anchors forced to Dx/Dy ≈ 0 at bottom edge to prevent unstable bumper lane-swinging.\n\n"
            "• 28-Element Observation (State) Vector:\n"
            "   - [Index 0]: Normalized Ego Velocity (v_ego / 30.0)\n"
            "   - [Index 1]: Normalized Steering Angle (bounded [-1.0, 1.0])\n"
            "   - [Indices 2-5]: 4 Lateral Lane Offsets (d₅, d₁₀, d₂₀, d₃₀) projected ahead\n"
            "   - [Indices 6-9]: 4 Heading Alignment Angles (θ₅, θ₁₀, θ₂₀, θ₃₀) projected ahead\n"
            "   - [Indices 10-23]: Geometric Lane Trajectory History (14 tracking points)\n"
            "   - [Indices 24-25]: LiDAR Safety States (front car distance & 12° forward corridor collision scan)\n"
            "   - [Indices 26-27]: Action Delay Compensation states (stacked past actions to overcome sim-to-real actuator lag)"
        )
    },
    {
        'id': 'slide_reward_shaping_s2r',
        'title': 'S2R-ACC: Physics-Informed Reward Function',
        'body': (
            "• Reward Structure: R_total = R_lane + R_speed - R_regularization\n\n"
            "• Lane Centering Reward (R_lane):\n"
            "   - R_lane = β * (1 - |center_distance|)^1.5\n"
            "   - Penalizes drift exponentially. β = 0.3 during multi-joint steering/speed training.\n\n"
            "• Velocity Tracking & Safety Reward (R_speed):\n"
            "   - Coupled with lane alignment: R_speed = (1 - β) * V_comp * (1 - |center_distance|)^1.5\n"
            "   - State 1 (Free Flow): Target speed is road limit (v_goal = v_limit)\n"
            "   - State 2 (Adaptive Following, d_front ≤ 15m): Set safe margin d_safe = v_ego * t_gap + d_min\n"
            "     * Too Close (d_front < d_safe) ➔ Force braking by setting v_goal = 0 m/s\n"
            "     * Safe Following (d_front ≥ d_safe) ➔ Match leader speed (v_goal = v_lead)\n"
            "   - Speed Component (V_comp): Gaussian bell curve e^(-(v_ego - v_goal)² / 2σ²) with continuous over-speed penalty.\n\n"
            "• Actuator Regularization (R_regularization):\n"
            "   - Steering Zig-Zag Penalty (P_steer): Scales up to 3x at 90 km/h to prevent dangerous micro-steering jitter.\n"
            "   - Lateral Sliding Penalty (P_lateral): Penalizes lateral velocity |v_lateral| to prevent aggressive weaving."
        )
    },
    {
        'id': 'slide_safety_guardrails_s2r',
        'title': 'S2R-ACC: Safe-Following & Guardrails',
        'body': (
            "• Safe-Following Queueing Reward (Stopped Loophole Solution):\n"
            "   - The Loophole: Severe stopped penalty normally terminates episode after 100 stopped steps (v < 0.1 m/s) on open road.\n"
            "   - The Solution: Identify safe queueing events behind a stopped front car (d_front < 12.0 m)\n"
            "     * Step-Level: Deactivate inactivity penalty; reward staying centered behind lead car with max(0.0, d_reward).\n"
            "     * Reset-Level: Terminate episode as a Clean Reset (R = 0, done = True) after 100 steps to avoid traffic penalties.\n\n"
            "• Safety Guardrails Summary:\n"
            "   - Lane Deviation (> 1.0 m) ➔ Episode Terminates ➔ Severe penalty (-100)\n"
            "   - Collision with Lead Car ➔ Episode Terminates ➔ Severe penalty (-100.0)\n"
            "   - Stopped on Open Road (no lead car) ➔ Episode Terminates after 100 steps ➔ Severe inactivity penalty\n"
            "   - Stopped in Traffic (d_front < 12.0 m) ➔ Clean Reset after 100 steps ➔ No penalty (0.0)"
        )
    }
]

def create_and_populate_slides():
    creds = get_credentials()
    service = get_slides_service(creds)
    
    # 1. Create the slides at index 1, 2, 3 (right after Slide 1)
    requests = []
    # Reverse order so that we insert slide_safety at index 1, slide_reward at index 1, slide_perception at index 1
    # This results in: Slide 1, Slide Perception (index 1), Slide Reward (index 2), Slide Safety (index 3)
    for slide_info in reversed(SLIDES_DATA):
        requests.append({
            'createSlide': {
                'objectId': slide_info['id'],
                'insertionIndex': 1,
                'slideLayoutReference': {
                    'predefinedLayout': 'TITLE_AND_BODY'
                }
            }
        })
        
    print("Creating 3 new slides after slide 1...")
    service.presentations().batchUpdate(
        presentationId=PRESENTATION_ID,
        body={'requests': requests}
    ).execute()
    
    # 2. Retrieve presentation to find the placeholder text box IDs
    print("Retrieving slide elements to locate text boxes...")
    presentation = service.presentations().get(presentationId=PRESENTATION_ID).execute()
    
    insert_text_requests = []
    for slide_data in SLIDES_DATA:
        slide_id = slide_data['id']
        title_text = slide_data['title']
        body_text = slide_data['body']
        
        title_id = None
        body_id = None
        
        # Search for the slide by ID
        found_slide = None
        for s in presentation.get('slides', []):
            if s.get('objectId') == slide_id:
                found_slide = s
                break
                
        if not found_slide:
            print(f"Error: Slide {slide_id} not found in presentation.")
            continue
            
        for element in found_slide.get('pageElements', []):
            if 'shape' in element and 'placeholder' in element['shape']:
                ph_type = element['shape']['placeholder'].get('type')
                if ph_type == 'TITLE':
                    title_id = element['objectId']
                elif ph_type == 'BODY':
                    body_id = element['objectId']
                    
        if title_id and body_id:
            insert_text_requests.extend([
                {
                    'insertText': {
                        'objectId': title_id,
                        'text': title_text
                    }
                },
                {
                    'insertText': {
                        'objectId': body_id,
                        'text': body_text
                    }
                }
            ])
        else:
            print(f"Error: Could not find placeholders on slide {slide_id}")
            
    if insert_text_requests:
        print("Populating titles and content in the 3 new slides...")
        service.presentations().batchUpdate(
            presentationId=PRESENTATION_ID,
            body={'requests': insert_text_requests}
        ).execute()
        print("Slides successfully created and populated!")
    else:
        print("No content updates to perform.")

if __name__ == '__main__':
    create_and_populate_slides()

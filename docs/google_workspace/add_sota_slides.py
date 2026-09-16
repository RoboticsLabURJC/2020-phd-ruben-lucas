#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from google_workspace.gapi_client import get_credentials, get_slides_service

PRESENTATION_ID = '1RHfwlYqdIFlw3tLAaJ_JH6rVCRP6D1NyMFhW667EiNY'

SLIDES_DATA = [
    {
        'id': 'slide_current_approach_eval_sota',
        'title': 'S2R-ACC: SOTA Current Approach Evaluation',
        'body': (
            "• Algorithm Selection:\n"
            "   - Soft Actor-Critic (SAC) is superior to on-policy PPO and classic DDPG for continuous control stability and sample efficiency, aligning with post-2024 SOTA.\n\n"
            "• Stopped Vehicle Loophole Solved:\n"
            "   - Custom Safe-Following Queueing Reward prevents traffic-lock/inactivity termination. Effectively models microscopic Intelligent Driver Model (IDM) queuing behaviors.\n\n"
            "• Sim-to-Real Jitter & Delay Mitigation:\n"
            "   - High-speed steering rate penalty (P_steer) and stacked past action delay compensation prevent physical controller chatter and lane swinging."
        )
    },
    {
        'id': 'slide_next_steps_sota',
        'title': 'S2R-ACC: SOTA Next Steps',
        'body': (
            "• Modular Decoupling / Hierarchical Control:\n"
            "   - Decouple lateral control (steering) from speed planning. Freeze pure steering w or use a classical geometry controller (MPC/Pure Pursuit), letting SAC focus 100% of its capacity on longitudinal ACC & queuing.\n\n"
            "• String Stability (Platooning Behavior):\n"
            "   - Simulate 3 to 5 CARLA vehicles in a platoon. Plot velocity profiles during sudden lead vehicle decelerations to prove braking wave attenuation (preventing rear-end shockwave amplification).\n\n"
            "• Standardized SOTA Benchmarking:\n"
            "   - Compare SAC performance directly against classical Intelligent Driver Model (IDM), Gipps' model, and DDPG/PPO baselines under FollowNet standards (Spacing MSE, Velocity MSE, TTC violation rate, and Jerk/Comfort)."
        )
    },
    {
        'id': 'slide_progression_roadmap_sota',
        'title': 'S2R-ACC: Thesis Progression Roadmap',
        'body': (
            "• [DONE] Integrated visual YOLOP perception with physics-informed safe ACC tracking.\n\n"
            "• [DONE] Solved stopped-traffic inactivity penalty loophole via Safe-Following Queueing.\n\n"
            "• [TODO] Decouple lateral steering (modular MPC/frozen agent) and train longitudinal-only SAC.\n\n"
            "• [TODO] Simulate CARLA platoon string stability under sudden lead vehicle braking.\n\n"
            "• [TODO] Quantify comparative benchmarking against IDM, Gipps, and DDPG baselines on Spacing MSE, TTC violation rate, and Jerk."
        )
    }
]

def create_and_populate_sota_slides():
    creds = get_credentials()
    service = get_slides_service(creds)
    
    # We want to insert these slides at index 4 (right after our safety slide which was inserted as Slide 4)
    # Reverse order so they end up in the correct order:
    # Index 4: Current Approach
    # Index 5: Next Steps
    # Index 6: Progression Roadmap
    requests = []
    for slide_info in reversed(SLIDES_DATA):
        requests.append({
            'createSlide': {
                'objectId': slide_info['id'],
                'insertionIndex': 4,
                'slideLayoutReference': {
                    'predefinedLayout': 'TITLE_AND_BODY'
                }
            }
        })
        
    print("Creating 3 new SOTA slides after Slide 4 (Index 4)...")
    service.presentations().batchUpdate(
        presentationId=PRESENTATION_ID,
        body={'requests': requests}
    ).execute()
    
    # Retrieve presentation to find placeholder text box IDs
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
        print("Populating titles and content in the 3 new SOTA slides...")
        service.presentations().batchUpdate(
            presentationId=PRESENTATION_ID,
            body={'requests': insert_text_requests}
        ).execute()
        print("SOTA slides successfully created and populated!")
    else:
        print("No content updates to perform.")

if __name__ == '__main__':
    create_and_populate_sota_slides()

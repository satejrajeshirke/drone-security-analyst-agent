# Real-image VLM results (gemini)

Command: `python -m drone_security_agent images data/real_frames --describer gemini`

| Image | Time / zone | VLM caption | Detections | Alerts |
|---|---|---|---|---|
| [03_person_at_gate.jpg](../data/real_frames/03_person_at_gate.jpg) | 00:05 main gate | At the main gate, one adult male security guard wearing a dark blue polo shirt, grey trousers, and a blue baseball cap stands beside a security booth next to a white traffic barrier, while a white sedan and a white SUV are parked in the background street. <br><sub>gemini-3.5-flash-lite (cached)</sub> | Guard (standing); White sedan (parked) | **MEDIUM** AFTER_HOURS_VEHICLE |
| [04_fence_climb.jpg](../data/real_frames/04_fence_climb.jpg) | 02:20 perimeter fence | At the perimeter fence, one man wearing a black jacket and grey trousers is climbing over while carrying a black briefcase, and another man wearing a dark jacket and jeans is climbing over a gate next to a green train with the text ЭР2-1213 on the front. <br><sub>gemini-3.5-flash-lite (cached)</sub> | Person (climbing) | **CRITICAL** PERIMETER_BREACH |
| [02_pickup_truck.jpg](../data/real_frames/02_pickup_truck.jpg) | 07:31 driveway | A white Dodge Ram 3500 pickup truck with license plate WOS-947 is parked in the driveway. <br><sub>gemini-3.5-flash-lite (cached)</sub> | White Dodge Ram (parked) | none |
| [05_loading_dock.jpg](../data/real_frames/05_loading_dock.jpg) | 08:10 loading dock | At the loading dock, a white Mercedes-Benz medium truck is parked and backing into the bay, alongside a white van visible in the background on the left. <br><sub>gemini-3.5-flash-lite (cached)</sub> | White Mercedes (parked); White van (loading) | none |
| [07_package_doorstep.jpg](../data/real_frames/07_package_doorstep.jpg) | 09:15 front door | Two people are at the front door: one woman with an afro is standing and carrying a gift box wrapped with a red ribbon and a canvas tote bag filled with packages, while another woman is standing and photographing her with a camera. <br><sub>gemini-3.5-flash-lite (cached)</sub> | 2 people (standing) | none |
| [01_parking_lot_aerial.jpg](../data/real_frames/01_parking_lot_aerial.jpg) | 13:20 parking lot | Clear view of the parking lot, no activity, containing about 1,000 parked cars. <br><sub>gemini-3.5-flash-lite (cached)</sub> | Car (parked) | none |
| [08_dog_in_yard.jpg](../data/real_frames/08_dog_in_yard.jpg) | 14:35 backyard | A small white dog is running across the backyard. <br><sub>gemini-3.5-flash-lite (cached)</sub> | Dog (running) | none |
| [06_night_people.jpg](../data/real_frames/06_night_people.jpg) | 23:15 warehouse | One person wearing dark clothing and a headlamp is standing on top of a hill at the warehouse. <br><sub>gemini-3.5-flash-lite (cached)</sub> | Person (standing) | **CRITICAL** RESTRICTED_ZONE |

VLM self-check: 0 'no activity' caption(s) re-checked, 0 corrected; VLM errors: 0; 8 caption(s) replayed from cache (out/vlm_cache.json; use --fresh to re-caption).

Image sources and licences: [ATTRIBUTION.md](../data/real_frames/ATTRIBUTION.md)

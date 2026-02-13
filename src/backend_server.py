import asyncio
import json
import websockets
import paho.mqtt.client as mqtt

# Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "vision/Phoenix_Team/#"
WS_PORT = 9002

connected_clients = set()

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    print(f"[MQTT] Received {msg.topic}: {payload}")
    
    # Broadcast to all connected WebSocket clients
    message = json.dumps({
        "topic": msg.topic,
        "payload": payload
    })
    
    # Schedule broadcasting in the asyncio loop
    asyncio.run_coroutine_threadsafe(broadcast_message(message), loop)

async def broadcast_message(message):
    if connected_clients:
        await asyncio.gather(*[client.send(message) for client in connected_clients], return_exceptions=True)

# WebSocket Handler
async def handler(websocket):
    print(f"[WS] Client connected: {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)
        print(f"[WS] Client disconnected: {websocket.remote_address}")

async def start_server():
    print(f"[WS] Starting server on port {WS_PORT}")
    async with websockets.serve(handler, "0.0.0.0", WS_PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    # Setup MQTT
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        print(f"[MQTT] Error connecting to broker: {e}")
        print("Ensure your MQTT Broker is running!")
        exit(1)

    # Setup Asyncio Loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(start_server())
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        client.loop_stop()

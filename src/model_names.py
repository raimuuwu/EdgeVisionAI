import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="10.140.123.226:8001")

models = client.get_model_repository_index()

print(models)

# models {
#   name: "boundary_detection"
#   version: "1"
#   state: "READY"
# }
# models {
#   name: "detection_preprocessing"
#   version: "1"
#   state: "READY"
# }
# models {
#   name: "ensemble_model"
#   version: "1"
#   state: "READY"
# }
# URL = "10.140.123.226:8001"  # IP Jetsona w sieci VPN
# MODEL_NAME = "boundary_detection"
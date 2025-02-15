from roboflow import Roboflow
rf = Roboflow(api_key="iW1QmBy39feV54Qtr575")
project = rf.workspace().project("banana-jtjak")
model = project.version(1).model
#
# # infer on a local image
print(model.predict("images.jpg", confidence=40, overlap=30).json())

# # visualize your prediction
model.predict("images.jpg", confidence=40, overlap=30).save("prediction2.jpg")

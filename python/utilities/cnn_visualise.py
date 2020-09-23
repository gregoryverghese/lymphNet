class GradCam:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName


    def findLayer(self):
        pass



    def heatmap(self, image, eps=K.epsilon()):

        gradModel = Model(inputs=[self.model.inputs],
                          outputs=[self.model.get_layer(self.layerName).output,
                                   self.model.output])

        with tf.GradientTape() as tape:

            inputs = tf.cast(image, tf.float32)
            c, predictions = gradModel(inputs)
            loss = predictions[:,self.classIdx]

        grads = tape.gradients(loss, c)

        castC = tf.cast(c>0, tf.float32)
        castGrads = tf.cast(grads > 0, tf.float32)
        guidedGrads = c * castC * castGrads

        castC = castC[0]
        guidedGrads = gudiedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0,1))
        cam = tf.reduce_sum(tf.multiply(weights, c), axis=-1)

        w, h = image.shape[0], image.shape[1]
        heatmap = cv2.resize(cam.numpy(), (w,h))

        numer = heatmap - np.min(heatmap)
        denom = (np.max(heatmap) - np.min(heatmap)) + eps
        heatmap = numer/denom
        heatmap = (heatmap*255).astype(np.uint8)

        return heatmap
    
        
        def overlay(self, heatMap, image, alpha=0.5, colormap = cv2.COLORMAP_VIRIDIS):

            heatmap = cv2.applyColorMap(heatmap, colormap)
            output = cv2.addWeighted(image alpha, heatmap, 1-alpha, 0)

            return heatmap, output






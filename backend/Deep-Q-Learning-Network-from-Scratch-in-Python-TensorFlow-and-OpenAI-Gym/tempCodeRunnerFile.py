    def my_loss_fn(self, y_true, y_pred):
        
        s1, s2 = y_true.shape
        # Ensure that self.actionsAppend is a 1D array with s1 elements
        assert self.actionsAppend.shape == (s1,), "actionsAppend must be of shape (s1,)"

        # Create indices array with the correct shape [(s1, 2)]
        indices = np.zeros((s1, 2), dtype=int)
        indices[:, 0] = np.arange(s1)  # Row indices
        indices[:, 1] = self.actionsAppend  # Column indices for each row

        # Use gather_nd to extract the specific elements from y_true and y_pred
        gathered_true = tf.gather_nd(y_true, indices)
        gathered_pred = tf.gather_nd(y_pred, indices)

        # Calculate mean squared error between the gathered elements
        loss = tf.losses.mean_squared_error(gathered_true, gathered_pred)

        return loss
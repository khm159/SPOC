import cv2, numpy as np, queue, threading

class ImageDisplayThread:
    def __init__(self, grid=(2, 2), tile_size=(320, 240), window_name="Dashboard"):
        self.rows, self.cols = grid
        self.tw, self.th = tile_size
        self.window = window_name
        self.feed_q = queue.Queue()
        self.running = True

        # Start GUI loop in main thread *after* constructing object
        threading.Thread(target=self._gui_loop, daemon=True).start()

    def show(self, slot_id: int, frame):
        """
        slot_id: 0..rows*cols-1  (row-major order)
        frame:   BGR image
        """
        if self.running:
            self.feed_q.put((slot_id, frame))

    def _gui_loop(self):
        # one big blank canvas
        H, W = self.th * self.rows, self.tw * self.cols
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        while self.running:
            # non-blocking fetch of latest frames
            try:
                while True:  # drain queue so we always display most recent frame per slot
                    slot, img = self.feed_q.get_nowait()
                    r, c = divmod(slot, self.cols)
                    x0, y0 = c * self.tw, r * self.th
                    img = cv2.resize(img, (self.tw, self.th))
                    canvas[y0:y0 + self.th, x0:x0 + self.tw] = img
            except queue.Empty:
                pass

            cv2.imshow(self.window, canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False

        cv2.destroyAllWindows()
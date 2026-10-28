import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


class VisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Практическая работа №3 — Контуры и ключевые точки")
        self.root.geometry("1200x760")
        self.original = None
        self.result = None
        self.original_photo = None
        self.result_photo = None

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")

        ttk.Button(top, text="Загрузить изображение", command=self.load_image).pack(side="left", padx=3)
        ttk.Button(top, text="Контуры Canny", command=self.show_canny).pack(side="left", padx=3)
        ttk.Button(top, text="Оператор Робертса", command=self.show_roberts).pack(side="left", padx=3)
        ttk.Button(top, text="Ключевые точки ORB", command=self.show_orb).pack(side="left", padx=3)
        ttk.Button(top, text="Найти номер", command=self.find_plate).pack(side="left", padx=3)

        cluster_frame = ttk.Frame(root, padding=(8, 0, 8, 8))
        cluster_frame.pack(fill="x")
        ttk.Label(cluster_frame, text="Количество кластеров K (2–20):").pack(side="left")
        self.k_var = tk.IntVar(value=5)
        self.k_spin = ttk.Spinbox(cluster_frame, from_=2, to=20, width=5, textvariable=self.k_var)
        self.k_spin.pack(side="left", padx=5)
        ttk.Button(cluster_frame, text="Сегментация K-Means", command=self.segment).pack(side="left", padx=3)
        ttk.Button(cluster_frame, text="Сброс результата", command=self.reset_result).pack(side="left", padx=3)

        images = ttk.Frame(root, padding=8)
        images.pack(fill="both", expand=True)
        left = ttk.LabelFrame(images, text="Исходное изображение", padding=5)
        right = ttk.LabelFrame(images, text="Результат обработки", padding=5)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        right.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self.original_label = ttk.Label(left, text="Загрузите изображение")
        self.original_label.pack(fill="both", expand=True)
        self.result_label = ttk.Label(right, text="Здесь появится результат")
        self.result_label.pack(fill="both", expand=True)

        self.status = tk.StringVar(value="Готово. Загрузите изображение.")
        ttk.Label(root, textvariable=self.status, anchor="w", padding=6).pack(fill="x")

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.webp"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Ошибка", "Не удалось открыть этот файл как изображение.")
            return
        self.original = image
        self.result = None
        self.display_image(self.original, self.original_label, "original")
        self.result_label.configure(image="", text="Здесь появится результат")
        self.result_photo = None
        self.status.set(f"Загружено: {path}")

    def display_image(self, image, label, target):
        if image is None:
            return
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail((540, 570), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_image)
        label.configure(image=photo, text="")
        label.image = photo
        if target == "original":
            self.original_photo = photo
        else:
            self.result_photo = photo

    def show_result(self, image, title):
        if self.original is None:
            messagebox.showwarning("Нет изображения", "Сначала загрузите изображение.")
            return
        self.result = image
        self.display_image(image, self.result_label, "result")
        self.status.set(title)

    def show_canny(self):
        if self.original is None:
            return self.need_image()
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 80, 160)
        self.show_result(edges, "Границы найдены детектором Canny.")

    def show_roberts(self):
        if self.original is None:
            return self.need_image()
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Ядра оператора Робертса 2x2
        gx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        gy = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        grad_x = cv2.filter2D(gray, cv2.CV_32F, gx)
        grad_y = cv2.filter2D(gray, cv2.CV_32F, gy)
        magnitude = cv2.magnitude(grad_x, grad_y)
        result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.show_result(result, "Границы найдены оператором Робертса.")

    def segment(self):
        if self.original is None:
            return self.need_image()
        try:
            k = int(self.k_var.get())
            if not 2 <= k <= 20:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Неверное значение", "Количество кластеров должно быть целым числом от 2 до 20.")
            return

        # Уменьшаем изображение для ускорения K-Means на больших фотографиях.
        h, w = self.original.shape[:2]
        scale = min(1.0, 350.0 / max(h, w))
        small = cv2.resize(self.original, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA) if scale < 1 else self.original.copy()
        pixels = np.float32(small.reshape((-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        cv2.setRNGSeed(42)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()].reshape(small.shape)
        segmented = cv2.resize(segmented, (w, h), interpolation=cv2.INTER_NEAREST)
        self.show_result(segmented, f"Сегментация K-Means выполнена: K = {k}.")

    def show_orb(self):
        if self.original is None:
            return self.need_image()
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=700)
        keypoints = orb.detect(gray, None)
        output = cv2.drawKeypoints(self.original, keypoints, None,
                                   color=(0, 255, 0),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.show_result(output, f"Ключевые точки ORB: найдено {len(keypoints)}.")

    def find_plate(self):
        if self.original is None:
            return self.need_image()
        # Эвристический поиск прямоугольных областей, похожих на номер.
        # Это демонстрационный метод, он не гарантирует обнаружение каждого номера.
        image = self.original.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(gray, 30, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]

        candidates = []
        ih, iw = gray.shape[:2]
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) != 4:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            if h == 0:
                continue
            ratio = w / float(h)
            area = w * h
            # Типичная форма номерной таблички: широкий прямоугольник.
            if 2.0 <= ratio <= 6.5 and area >= (iw * ih) * 0.001 and w > 60 and h > 15:
                candidates.append((area, x, y, w, h))

        if candidates:
            _, x, y, w, h = max(candidates, key=lambda item: item[0])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(image, "Possible license plate", (x, max(25, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            self.show_result(image, "Найдена возможная область автомобильного номера (эвристика).")
        else:
            self.show_result(image, "Подходящий прямоугольник не найден. Попробуйте фото с чётко видимым номером.")

    def reset_result(self):
        if self.original is None:
            return self.need_image()
        self.show_result(self.original.copy(), "Результат сброшен: показана копия исходного изображения.")

    def need_image(self):
        messagebox.showinfo("Сначала загрузите изображение", "Нажмите «Загрузить изображение» и выберите фото.")


if __name__ == "__main__":
    root = tk.Tk()
    app = VisionApp(root)
    root.mainloop()

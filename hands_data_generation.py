import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ============================================================
#  GESTURE COLLECTOR (LOGIC)
# ============================================================
class GestureCollector:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.no_of_frames = 500
        self.is_collecting = False
        self.should_stop = False
        self.all_labels = ['neutral'] + [chr(i) for i in range(65, 91)]

    def make_landmark_timestamp(self, results):
        """21 titik × (x,y,z) = 63 fitur"""
        cLm = []
        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            for lm in hand_lms.landmark:
                cLm.extend([lm.x, lm.y, lm.z])
        return cLm + [0] * (63 - len(cLm))

    def collect_data(self, labels_to_collect, progress_callback,
                     status_callback, stats_callback):
        """Main collection loop"""
        CAMERA_INDEX = 0
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(3, 640)
        cap.set(4, 480)

        if not cap.isOpened():
            status_callback("❌ CAMERA TIDAK TERDETEKSI!", "error")
            return

        status_callback("✅ Kamera siap digunakan", "success")
        stats_callback(total_gesture=len(labels_to_collect))

        total_frames_captured = 0

        for idx, label in enumerate(labels_to_collect):
            if self.should_stop:
                break

            lmlist = []
            status_callback(
                f"📹 Siap merekam gesture '{label}' (SPACE untuk mulai)",
                "info"
            )

            # -------- MENUNGGU SPACE --------
            waiting = True
            while waiting and not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)

                cv2.putText(
                    frame, f"Gesture: {label.upper()}",
                    (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.2, (0, 255, 255), 2
                )
                cv2.putText(
                    frame, "SPACE: Mulai | Q: Keluar",
                    (20, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2
                )

                if label == 'neutral':
                    tip = "Tangan rileks, telapak terbuka"
                else:
                    tip = f"Pose huruf {label.upper()}"
                cv2.putText(
                    frame, tip,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2
                )

                cv2.imshow("Gesture Data Collection", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord(' '):
                    status_callback(f"🎬 Recording '{label}'...", "recording")
                    waiting = False
                if k == ord('q'):
                    self.should_stop = True
                    waiting = False

            if self.should_stop:
                break

            # -------- RECORDING --------
            with self.mpHands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            ) as hands:

                while len(lmlist) < self.no_of_frames and not self.should_stop:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frameRGB)

                    if results.multi_hand_landmarks:
                        self.mpDraw.draw_landmarks(
                            frame,
                            results.multi_hand_landmarks[0],
                            self.mpHands.HAND_CONNECTIONS
                        )
                        lm = self.make_landmark_timestamp(results)
                        lmlist.append(lm)

                        progress = len(lmlist) / self.no_of_frames * 100

                        bar_length = 400
                        filled_length = int(
                            bar_length * len(lmlist) / self.no_of_frames
                        )
                        cv2.rectangle(
                            frame, (20, 80),
                            (20 + bar_length, 110),
                            (50, 50, 50), -1
                        )
                        cv2.rectangle(
                            frame, (20, 80),
                            (20 + filled_length, 110),
                            (0, 255, 0), -1
                        )

                        cv2.putText(
                            frame,
                            f"{label.upper()}: {len(lmlist)}/"
                            f"{self.no_of_frames} ({progress:.1f}%)",
                            (20, 60),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.9, (0, 255, 0), 2
                        )

                        progress_callback(idx, len(labels_to_collect),
                                          progress)
                    else:
                        cv2.putText(
                            frame,
                            "TANGAN TIDAK TERDETEKSI!",
                            (20, 60),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.9, (0, 0, 255), 2
                        )

                    cv2.putText(
                        frame,
                        f"Gesture: {label.upper()}",
                        (20, frame.shape[0] - 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2
                    )
                    cv2.putText(
                        frame,
                        "JANGAN UBAH POSISI! | Q=Quit",
                        (20, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2
                    )

                    cv2.imshow("Gesture Data Collection", frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'):
                        self.should_stop = True
                        break

            # -------- SIMPAN DATA --------
            if len(lmlist) > 0:
                df = pd.DataFrame(lmlist)
                filename = f"{label}.txt"
                df.to_csv(filename, index=False)
                total_frames_captured += len(lmlist)

                status_callback(
                    f"✅ {filename} disimpan ({len(lmlist)} frame)",
                    "success"
                )

                stats_callback(
                    captured_frames=total_frames_captured,
                    completed_gesture=idx + 1,
                    last_saved=filename
                )

            time.sleep(0.5)

        cap.release()
        cv2.destroyAllWindows()

        if not self.should_stop:
            status_callback("🎉 Semua data berhasil dikumpulkan!", "complete")
        else:
            status_callback("⚠️ Collection dihentikan", "warning")


# ============================================================
#  MODERN TKINTER UI
# ============================================================
class GestureCollectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Data Collection - Training")
        self.root.geometry("1280x720")
        self.root.minsize(1100, 650)

        self.root.configure(bg="#20252B")
        self.root.bind('<Escape>', lambda e: self.toggle_fullscreen())
        self.root.bind('<F11>', lambda e: self.toggle_fullscreen())
        self.is_fullscreen = False

        # ----- ttk style -----
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("TFrame", background="#20252B")
        style.configure("Card.TFrame", background="#252B33", relief="flat")
        style.configure("Title.TLabel", background="#1F2933",
                        foreground="white", font=("Segoe UI", 18, "bold"))
        style.configure("Section.TLabelframe", background="#20252B",
                        foreground="white", font=("Segoe UI", 11, "bold"))
        style.configure("Section.TLabelframe.Label", background="#20252B",
                        foreground="#E5E7EB", font=("Segoe UI", 11, "bold"))
        style.configure("TLabel", background="#20252B",
                        foreground="#E5E7EB")
        style.configure("Status.TLabel", background="#111827",
                        foreground="#E5E7EB", font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", background="#10B981",
                        foreground="white", font=("Segoe UI", 11, "bold"))
        style.map("Accent.TButton",
                  background=[("active", "#059669")])
        style.configure("Danger.TButton", background="#EF4444",
                        foreground="white", font=("Segoe UI", 11, "bold"))
        style.map("Danger.TButton",
                  background=[("active", "#DC2626")])

        self.collector = GestureCollector()
        self.selected_labels = []
        self.total_gesture = 0
        self.completed_gesture = 0
        self.total_frames = 0

        self.build_layout()
        self.apply_range(initial=True)

    # --------------------------------------------------------
    #  Layout utama
    # --------------------------------------------------------
    def build_layout(self):
        # TOP BAR
        top_bar = tk.Frame(self.root, bg="#1F2933", height=56)
        top_bar.pack(side=tk.TOP, fill=tk.X)
        top_bar.pack_propagate(False)

        left_title = tk.Frame(top_bar, bg="#1F2933")
        left_title.pack(side=tk.LEFT, padx=20)
        icon = tk.Label(left_title, text="🖐️",
                        bg="#1F2933", fg="#10B981",
                        font=("Segoe UI Emoji", 20))
        icon.pack(side=tk.LEFT, pady=10)
        title_label = ttk.Label(left_title,
                                text="Gesture Data Collection",
                                style="Title.TLabel")
        title_label.pack(side=tk.LEFT, padx=10)

        right_top = tk.Frame(top_bar, bg="#1F2933")
        right_top.pack(side=tk.RIGHT, padx=10)
        self.time_label = tk.Label(right_top, text="",
                                   bg="#1F2933",
                                   fg="#9CA3AF",
                                   font=("Segoe UI", 9))
        self.time_label.pack(side=tk.TOP, anchor="e")
        exit_btn = tk.Button(
            right_top,
            text="✕ Exit (ESC)",
            command=self.root.quit,
            bg="#EF4444",
            fg="white",
            bd=0,
            padx=14,
            pady=6,
            font=("Segoe UI", 10, "bold"),
            activebackground="#DC2626",
            activeforeground="white",
            cursor="hand2"
        )
        exit_btn.pack(side=tk.BOTTOM, pady=4, anchor="e")
        self.update_clock()

        # MAIN AREA
        main = tk.Frame(self.root, bg="#20252B")
        main.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(main, bg="#111827", width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        content = tk.Frame(main, bg="#20252B")
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.build_sidebar(sidebar)
        self.build_content(content)

        # STATUS BAR
        status_bar = tk.Frame(self.root, bg="#111827", height=28)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        status_bar.pack_propagate(False)

        self.status_label = ttk.Label(
            status_bar,
            text="Status: Siap",
            style="Status.TLabel",
            anchor="w"
        )
        self.status_label.pack(side=tk.LEFT, padx=10,
                               fill=tk.X, expand=True)

        self.right_status = tk.Label(
            status_bar,
            text="Ready",
            bg="#111827",
            fg="#9CA3AF",
            font=("Segoe UI", 9)
        )
        self.right_status.pack(side=tk.RIGHT, padx=10)

    # --------------------------------------------------------
    #  Sidebar (kiri)
    # --------------------------------------------------------
    def build_sidebar(self, parent):
        header = tk.Frame(parent, bg="#111827")
        header.pack(fill=tk.X, padx=16, pady=(18, 6))
        tk.Label(
            header,
            text="Setup Collection",
            bg="#111827",
            fg="#F9FAFB",
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Atur range huruf dan opsi gesture.",
            bg="#111827",
            fg="#6B7280",
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(2, 0))

        select_frame = ttk.LabelFrame(
            parent, text="Pilih Gesture",
            style="Section.TLabelframe"
        )
        select_frame.pack(fill=tk.X, padx=14,
                          pady=(6, 10), ipadx=4, ipady=4)

        range_frame = tk.Frame(select_frame, bg="#20252B")
        range_frame.pack(fill=tk.X, pady=6)

        tk.Label(
            range_frame, text="Dari",
            bg="#20252B", fg="#E5E7EB",
            font=("Segoe UI", 10)
        ).grid(row=0, column=0, padx=(4, 4), pady=4, sticky="w")
        self.from_var = tk.StringVar(value="A")
        self.from_combo = ttk.Combobox(
            range_frame,
            textvariable=self.from_var,
            width=5, state="readonly",
            font=("Segoe UI", 10)
        )
        self.from_combo['values'] = [chr(i) for i in range(65, 91)]
        self.from_combo.grid(row=0, column=1,
                             padx=(0, 10), pady=4, sticky="w")

        tk.Label(
            range_frame, text="Sampai",
            bg="#20252B", fg="#E5E7EB",
            font=("Segoe UI", 10)
        ).grid(row=1, column=0, padx=(4, 4), pady=4, sticky="w")
        self.to_var = tk.StringVar(value="Z")
        self.to_combo = ttk.Combobox(
            range_frame,
            textvariable=self.to_var,
            width=5, state="readonly",
            font=("Segoe UI", 10)
        )
        self.to_combo['values'] = [chr(i) for i in range(65, 91)]
        self.to_combo.grid(row=1, column=1,
                           padx=(0, 10), pady=4, sticky="w")

        self.neutral_var = tk.BooleanVar(value=True)
        neutral_cb = tk.Checkbutton(
            select_frame,
            text="Include gesture 'neutral'",
            variable=self.neutral_var,
            bg="#20252B",
            fg="#E5E7EB",
            activebackground="#20252B",
            activeforeground="#FFFFFF",
            selectcolor="#111827",
            font=("Segoe UI", 9)
        )
        neutral_cb.pack(anchor="w", padx=8, pady=(4, 4))

        apply_btn = tk.Button(
            select_frame,
            text="✓ Terapkan Range",
            command=self.apply_range,
            bg="#2563EB",
            fg="white",
            bd=0,
            padx=10,
            pady=6,
            font=("Segoe UI", 10, "bold"),
            activebackground="#1D4ED8",
            activeforeground="white",
            cursor="hand2"
        )
        apply_btn.pack(fill=tk.X, padx=6, pady=(4, 6))

        self.selected_text = tk.Text(
            select_frame,
            height=6,
            font=("Consolas", 9),
            bg="#111827",
            fg="#E5E7EB",
            relief="flat",
            wrap="word"
        )
        self.selected_text.pack(fill=tk.BOTH, padx=6, pady=(0, 4))
        self.selected_text.insert("1.0", "Belum ada gesture yang dipilih")
        self.selected_text.config(state=tk.DISABLED)

        option_frame = ttk.LabelFrame(
            parent, text="Opsi Tambahan",
            style="Section.TLabelframe"
        )
        option_frame.pack(fill=tk.X, padx=14,
                          pady=(0, 10), ipadx=4, ipady=4)

        self.frames_var = tk.IntVar(value=self.collector.no_of_frames)
        tk.Label(
            option_frame,
            text="Jumlah frame / gesture",
            font=("Segoe UI", 9),
            background="#20252B"
        ).pack(anchor="w", padx=8, pady=(2, 0))
        self.frames_entry = ttk.Entry(
            option_frame,
            textvariable=self.frames_var,
            width=10
        )
        self.frames_entry.pack(anchor="w", padx=8, pady=(0, 6))

        btn_frame = tk.Frame(option_frame, bg="#20252B")
        btn_frame.pack(fill=tk.X, padx=4, pady=(2, 2))

        self.start_btn = ttk.Button(
            btn_frame,
            text="▶  START",
            style="Accent.TButton",
            command=self.start_collection
        )
        self.start_btn.pack(side=tk.LEFT, fill=tk.X,
                            expand=True, padx=(2, 4))

        self.stop_btn = ttk.Button(
            btn_frame,
            text="⏹  STOP",
            style="Danger.TButton",
            command=self.stop_collection,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X,
                           expand=True, padx=(4, 2))

        hint = tk.Label(
            parent,
            text="Tips:\n- Pastikan tangan berada di tengah frame.\n"
                 "- Gunakan background yang kontras.\n"
                 "- Jaga jarak kamera konstan.",
            bg="#111827",
            fg="#9CA3AF",
            justify="left",
            font=("Segoe UI", 8)
        )
        hint.pack(fill=tk.X, padx=16, pady=(2, 10))

    # --------------------------------------------------------
    #  Content (kanan)
    # --------------------------------------------------------
    def build_content(self, parent):
        # Kartu ringkasan
        cards_frame = tk.Frame(parent, bg="#20252B")
        cards_frame.pack(fill=tk.X, padx=18, pady=(16, 8))

        card1 = ttk.Frame(cards_frame, style="Card.TFrame")
        card1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        tk.Label(
            card1, text="Total Gesture",
            bg="#252B33", fg="#9CA3AF",
            font=("Segoe UI", 9)
        ).pack(anchor="w", padx=10, pady=(8, 0))
        self.card_total_gesture = tk.Label(
            card1, text="0",
            bg="#252B33", fg="#F9FAFB",
            font=("Segoe UI", 18, "bold")
        )
        self.card_total_gesture.pack(anchor="w", padx=10, pady=(0, 8))

        card2 = ttk.Frame(cards_frame, style="Card.TFrame")
        card2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        tk.Label(
            card2, text="Gesture Selesai",
            bg="#252B33", fg="#9CA3AF",
            font=("Segoe UI", 9)
        ).pack(anchor="w", padx=10, pady=(8, 0))
        self.card_completed = tk.Label(
            card2, text="0",
            bg="#252B33", fg="#F9FAFB",
            font=("Segoe UI", 18, "bold")
        )
        self.card_completed.pack(anchor="w", padx=10, pady=(0, 8))

        card3 = ttk.Frame(cards_frame, style="Card.TFrame")
        card3.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        tk.Label(
            card3, text="Total Frame",
            bg="#252B33", fg="#9CA3AF",
            font=("Segoe UI", 9)
        ).pack(anchor="w", padx=10, pady=(8, 0))
        self.card_frames = tk.Label(
            card3, text="0",
            bg="#252B33", fg="#F9FAFB",
            font=("Segoe UI", 18, "bold")
        )
        self.card_frames.pack(anchor="w", padx=10, pady=(0, 8))

        middle = tk.Frame(parent, bg="#20252B")
        middle.pack(fill=tk.BOTH, expand=True,
                    padx=18, pady=(0, 12))

        progress_frame = ttk.LabelFrame(
            middle, text="Progress",
            style="Section.TLabelframe"
        )
        progress_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=(10, 4))

        self.progress_label = tk.Label(
            progress_frame,
            text="0%",
            bg="#20252B",
            fg="#E5E7EB",
            font=("Segoe UI", 9)
        )
        self.progress_label.pack(anchor="w", padx=12, pady=(0, 6))

        log_frame = ttk.LabelFrame(
            middle, text="Activity Log",
            style="Section.TLabelframe"
        )
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH,
                       expand=True, pady=(0, 0))

        inner = tk.Frame(log_frame, bg="#20252B")
        inner.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.log_text = tk.Text(
            inner,
            height=10,
            font=("Consolas", 9),
            bg="#111827",
            fg="#D1D5DB",
            relief="flat",
            wrap="word"
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(
            inner, orient="vertical",
            command=self.log_text.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

    # --------------------------------------------------------
    #  Helper / event
    # --------------------------------------------------------
    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes('-fullscreen', self.is_fullscreen)

    def update_clock(self):
        now = datetime.now().strftime("%d %b %Y  %H:%M:%S")
        self.time_label.config(text=now)
        self.root.after(1000, self.update_clock)

    def apply_range(self, initial=False):
        from_letter = self.from_var.get()
        to_letter = self.to_var.get()
        from_ord = ord(from_letter)
        to_ord = ord(to_letter)

        if from_ord > to_ord and not initial:
            messagebox.showerror("Error",
                                 "Huruf 'Dari' harus <= 'Sampai'")
            return

        if from_ord > to_ord:
            from_ord, to_ord = to_ord, from_ord

        selected = [chr(i) for i in range(from_ord, to_ord + 1)]

        if self.neutral_var.get():
            selected = ['neutral'] + selected

        self.selected_labels = selected
        self.total_gesture = len(selected)
        self.completed_gesture = 0

        self.update_cards()
        self.selected_text.config(state=tk.NORMAL)
        self.selected_text.delete("1.0", tk.END)
        self.selected_text.insert(
            "1.0",
            f"Gesture terpilih ({len(selected)}):\n"
            + ", ".join(selected)
        )
        self.selected_text.config(state=tk.DISABLED)

        self.log(f"✅ {len(selected)} gesture dipilih: "
                 f"{', '.join(selected)}")

    def update_cards(self, last_saved=None):
        self.card_total_gesture.config(text=str(self.total_gesture))
        self.card_completed.config(text=str(self.completed_gesture))
        self.card_frames.config(text=str(self.total_frames))
        if last_saved:
            self.right_status.config(
                text=f"Last file: {last_saved}"
            )

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(
            tk.END, f"[{timestamp}] {message}\n"
        )
        self.log_text.see(tk.END)

    def update_progress(self, current_idx, total, gesture_progress):
        overall_progress = (current_idx / total) * 100 \
            + (gesture_progress / total)
        self.progress_bar['value'] = overall_progress
        self.progress_label.config(
            text=f"{overall_progress:.1f}%"
        )

    def update_status(self, message, status_type="info"):
        colors = {
            "info": "#60A5FA",
            "success": "#10B981",
            "error": "#F87171",
            "warning": "#F59E0B",
            "recording": "#C4B5FD",
            "complete": "#22C55E"
        }
        color = colors.get(status_type, "#E5E7EB")
        self.status_label.config(
            text=f"Status: {message}",
            foreground=color
        )
        self.log(message)

    def update_stats(self, total_gesture=None,
                     captured_frames=None,
                     completed_gesture=None,
                     last_saved=None):
        if total_gesture is not None:
            self.total_gesture = total_gesture
        if captured_frames is not None:
            self.total_frames = captured_frames
        if completed_gesture is not None:
            self.completed_gesture = completed_gesture
        self.update_cards(last_saved)

    def start_collection(self):
        if not self.selected_labels:
            messagebox.showwarning(
                "Warning", "Pilih gesture terlebih dahulu!"
            )
            return

        try:
            nf = int(self.frames_var.get())
            if nf <= 0:
                raise ValueError
            self.collector.no_of_frames = nf
        except Exception:
            messagebox.showerror(
                "Error",
                "Jumlah frame harus bilangan bulat positif!"
            )
            return

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.collector.should_stop = False
        self.progress_bar['value'] = 0
        self.progress_label.config(text="0%")

        self.log("\n" + "=" * 60)
        self.log("🎬 MEMULAI COLLECTION...")
        self.log(f"Total gesture: {len(self.selected_labels)}")
        self.log("=" * 60 + "\n")

        thread = threading.Thread(
            target=self.collector.collect_data,
            args=(
                self.selected_labels,
                self.update_progress,
                self.update_status,
                self.update_stats
            ),
            daemon=True
        )
        thread.start()
        self.monitor_thread(thread)

    def monitor_thread(self, thread):
        if thread.is_alive():
            self.root.after(
                150, lambda: self.monitor_thread(thread)
            )
        else:
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def stop_collection(self):
        self.collector.should_stop = True
        self.log("⏹ Menghentikan collection...")
        self.stop_btn.config(state=tk.DISABLED)


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = GestureCollectorGUI(root)
    root.mainloop()

    def _build_ui(self):
        """Tạo UI chính (2 màn: chọn nguồn & main)."""
      
        self.select_frame = tk.Frame(self.root, bg="#f2e9e4")
        self.select_frame.pack(fill="both", expand=True)

        title = tk.Label(self.select_frame, text="Chọn nguồn camera", bg="#f2e9e4", font=("Arial", 18, "bold"))
        title.pack(pady=(30, 10))

        top_frame = tk.Frame(self.select_frame, bg="#f2e9e4")
        top_frame.pack(pady=(15, 5))

        tk.Label(top_frame, text="Nguồn camera:", bg="#f2e9e4", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=(0, 10))
        self.cam_var = tk.StringVar(value="0")

        rb_internal = tk.Radiobutton(top_frame, text="Camera máy tính (0)", variable=self.cam_var, value="0", bg="#f2e9e4", font=("Arial", 11))
        rb_external = tk.Radiobutton(top_frame, text="Camera rời (1)", variable=self.cam_var, value="1", bg="#f2e9e4", font=("Arial", 11))
        rb_internal.grid(row=0, column=1, padx=10)
        rb_external.grid(row=0, column=2, padx=10)

        apply_btn = tk.Button(self.select_frame, text="Áp dụng nguồn", command=self.apply_camera_choice_and_go,
                              bg="#ffffff", activebackground="#fde2e4", relief="raised", bd=2, font=("Arial", 12, "bold"), cursor="hand2")
        apply_btn.pack(pady=20)

        # --- Main frame ---
        self.main_frame = tk.Frame(self.root, bg="#f2e9e4")

        # Top bar
        top_bar = tk.Frame(self.main_frame, bg="#f2e9e4")
        top_bar.pack(pady=(10, 0), fill="x")

        self.back_btn = tk.Button(top_bar, text="⟵ Quay lại chọn nguồn", command=self.go_back_to_select,
                                  bg="#ffffff", activebackground="#fde2e4", relief="raised", bd=2, font=("Arial", 11, "bold"), cursor="hand2")
        self.back_btn.pack(side="left", padx=20)

        # Checkbuttons
        self.student_detection_cb = tk.Checkbutton(top_bar, text="Nhận diện học sinh", variable=self.student_detection_enabled,
                                                   bg="#f2e9e4", font=("Arial", 11, "bold"), command=self.toggle_student_detection)
        self.student_detection_cb.pack(side="right", padx=10)
        if self.student_detection_enabled.get():
            self.student_detection_cb.select()

        self.detection_cb = tk.Checkbutton(top_bar, text="Nhận diện hành vi", variable=self.detection_enabled,
                                           bg="#f2e9e4", font=("Arial", 11, "bold"), command=self.toggle_detection)
        self.detection_cb.pack(side="right", padx=10)
        if self.detection_enabled.get():
            self.detection_cb.select()
            
        # Thêm nút quản lý học sinh vào top_bar
        self.manage_students_btn = tk.Button(top_bar, text="Quản lý học sinh", command=self.manage_student_list,
                                           bg="#ffffff", activebackground="#fde2e4", relief="raised", bd=2, 
                                           font=("Arial", 11, "bold"), cursor="hand2")
        self.manage_students_btn.pack(side="right", padx=10)
        
        # Thêm nút cấu hình vào top_bar
        self.settings_btn = tk.Button(top_bar, text="⚙️ Cấu hình", command=self.show_settings,
                                    bg="#ffffff", activebackground="#fde2e4", relief="raised", bd=2, 
                                    font=("Arial", 11, "bold"), cursor="hand2")
        self.settings_btn.pack(side="right", padx=10)
        
        # Thêm nút huấn luyện mô hình vào top_bar
        self.train_model_btn = tk.Button(top_bar, text="🧠 Huấn luyện mô hình", command=self.train_student_model,
                                       bg="#ffffff", activebackground="#fde2e4", relief="raised", bd=2, 
                                       font=("Arial", 11, "bold"), cursor="hand2")
        self.train_model_btn.pack(side="right", padx=10)
        
        # Thêm nút thu thập dữ liệu vào top_bar
        self.collect_data_btn = tk.Button(top_bar, text="📸 Thu thập dữ liệu", command=self.collect_face_data,
                                        bg="#ffffff", activebackground="#fde2e4", relief="raised", bd=2, 
                                        font=("Arial", 11, "bold"), cursor="hand2")
        self.collect_data_btn.pack(side="right", padx=10)

        # Buttons frame
        btn_frame = tk.Frame(self.main_frame, bg="#f2e9e4")
        btn_frame.pack(pady=12, fill="x")

        btn_style = {"width": 14, "height": 3, "compound": "top",
                     "bg": "#fff", "activebackground": "#fde2e4",
                     "relief": "raised", "bd": 2, "font": ("Arial", 11, "bold"),
                     "cursor": "hand2"}

        # Icons load (improve resilience)
        start_icon = self._safe_load_icon("start.png", (64, 64))
        stop_icon = self._safe_load_icon("stop.png", (64, 64))
        report_icon = self._safe_load_icon("report.png", (64, 64))

        self.start_btn = tk.Button(btn_frame, text="Bắt đầu", image=start_icon, command=self.start_counting, **btn_style)
        self.start_btn.image = start_icon
        self.start_btn.pack(side="left", padx=40)

        self.stop_btn = tk.Button(btn_frame, text="Dừng", image=stop_icon, command=self.stop_counting, **btn_style)
        self.stop_btn.image = stop_icon
        self.stop_btn.pack(side="left", padx=40)

        self.report_btn = tk.Button(btn_frame, text="Báo cáo", image=report_icon, command=self.show_behavior_report, **btn_style)
        self.report_btn.image = report_icon
        self.report_btn.pack(side="left", padx=40)

        # Video display
        self.video_label = tk.Label(self.main_frame, bg="#c9ada7")
        self.video_label.pack(pady=10, fill="both", expand=True)

        # Count frame
        self.count_frame = tk.Frame(self.main_frame, bg="#f2e9e4")
        self.count_frame.pack(pady=5, fill="x")

        self.count_label = tk.Label(self.count_frame, text="Đang chờ bắt đầu đếm...", bg="#f2e9e4", font=("Arial", 12), fg="#2d3436")
        self.count_label.pack(pady=5)

        self.info_label = tk.Label(self.main_frame, text="🟡 Chưa mở camera. Hãy quay lại chọn nguồn và áp dụng.",
                                   bg="#f2e9e4", font=("Arial", 14, "bold"), fg="#2d3436")
        self.info_label.pack(pady=8)

    def _safe_load_icon(self, name: str, size: Tuple[int, int]) -> Optional[ImageTk.PhotoImage]:
        """Tải icon nếu có, trả về None nếu lỗi (không raise)."""
        try:
            path = self.resource_path(name)
            img = Image.open(path).resize(size)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    def init_camera(self, index: int) -> bool:
        """Khởi tạo camera - trả về True nếu thành công."""
        try:
            if self.cap and getattr(self.cap, "isOpened", lambda: False)():
                try:
                    self.cap.release()
                except Exception:
                    pass
            # Chọn backend phù hợp
            if sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(index)
            ok = self.cap.isOpened()
            if ok:
                self.current_cam_index = index
                LOG.info("Mở camera %s thành công", index)
            else:
                self.current_cam_index = None
                LOG.warning("Không mở được camera %s", index)
            return ok
        except Exception as e:
            LOG.exception("Lỗi init_camera: %s", e)
            return False

    def apply_camera_choice_and_go(self):
        """Xử lý khi ấn 'Áp dụng nguồn'."""
        if self.counting:
            messagebox.showwarning("Đang đếm", "Vui lòng dừng đếm trước khi đổi nguồn.")
            return

        cam_index = int(self.cam_var.get())
        if not self.init_camera(cam_index):
            alt = 1 - cam_index
            if self.init_camera(alt):
                self.cam_var.set(str(alt))
                messagebox.showwarning("Chú ý", f"Không mở được camera {cam_index}. Đã thử chuyển sang {alt}.")
            else:
                messagebox.showerror("Lỗi", f"Không thể mở camera {cam_index} và {alt}.")
                return

        self.show_main_ui()
        status = f"🟢 Camera {self.current_cam_index} đang bật"
        if self.student_detection_enabled.get() and self.detection_enabled.get():
            status += " với nhận diện học sinh và hành vi đang hoạt động"
        elif self.student_detection_enabled.get():
            status += " với nhận diện học sinh đang hoạt động"
        elif self.detection_enabled.get():
            status += " với nhận diện hành vi đang hoạt động"
        else:
            status += " – sẵn sàng đếm hành vi."
        self.info_label.config(text=status)

    def go_back_to_select(self):
        if self.counting:
            messagebox.showwarning("Đang đếm", "Hãy dừng đếm trước khi quay lại màn chọn nguồn.")
            return
        self.main_frame.pack_forget()
        self.select_frame.pack(fill="both", expand=True)

    def show_main_ui(self):
        self.select_frame.pack_forget()
        self.main_frame.pack(fill="both", expand=True)
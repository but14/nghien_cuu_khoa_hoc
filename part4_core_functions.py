    def toggle_detection(self):
        if self.detection_enabled.get():
            if not (self.torch_available or self.ultralytics_available):
                self.detection_enabled.set(False)
                messagebox.showerror("Lỗi", "Thiếu thư viện để bật nhận diện hành vi: pip install torch ultralytics")
                return
            if self.model is None and not self.load_model():
                self.detection_enabled.set(False)
                return
            messagebox.showinfo("Thông báo", "Đã bật nhận diện hành vi.")
        else:
            messagebox.showinfo("Thông báo", "Đã tắt nhận diện hành vi.")

    def toggle_student_detection(self):
        if self.student_detection_enabled.get():
            if not self.torch_available:
                self.student_detection_enabled.set(False)
                messagebox.showerror("Lỗi", "Thiếu torch: pip install torch")
                return
            if self.student_model is None and not self.load_student_model():
                self.student_detection_enabled.set(False)
                return
            messagebox.showinfo("Thông báo", "Đã bật nhận diện học sinh.")
        else:
            self.current_student = None
            messagebox.showinfo("Thông báo", "Đã tắt nhận diện học sinh.")

    def start_counting(self):
        if not (self.cap and getattr(self.cap, "isOpened", lambda: False)()):
            messagebox.showerror("Lỗi", "Chưa có camera nào đang mở.")
            return
        if not self.detection_enabled.get():
            messagebox.showerror("Lỗi", "Vui lòng bật chế độ nhận diện hành vi trước khi đếm.")
            return
        if self.counting:
            messagebox.showinfo("Thông báo", "Đã đang đếm.")
            return

        # Reset counters
        self.behavior_counts = Counter()
        self.student_behavior_counts = defaultdict(Counter)
        self.current_behaviors = {}
        self.last_detected = None
        self.last_detected_time = 0.0

        self.counting = True
        self.start_time = time.time()
        self.recorded_time = 0
        if self.student_detection_enabled.get():
            status = f"🔴 Đang đếm hành vi theo học sinh (camera {self.current_cam_index})..."
            if self.current_student:
                self.count_label.config(text=f"Đang đếm hành vi cho học sinh: {self.current_student}...\nChưa phát hiện hành vi nào.")
            else:
                self.count_label.config(text="Đang đếm hành vi...\nĐang đợi nhận diện học sinh.")
            messagebox.showinfo("Thông báo", "Bắt đầu đếm hành vi theo học sinh!")
        else:
            status = f"🔴 Đang đếm hành vi (camera {self.current_cam_index})..."
            self.count_label.config(text="Đang đếm hành vi...\nChưa phát hiện hành vi nào.")
            messagebox.showinfo("Thông báo", "Bắt đầu đếm hành vi!")

        self.info_label.config(text=status)

    def stop_counting(self):
        if not self.counting:
            messagebox.showinfo("Thông báo", "Chưa bắt đầu đếm hành vi.")
            return
        self.counting = False
        self.recorded_time = int(time.time() - self.start_time) if self.start_time else 0
        status = f"🟡 Đã dừng đếm. Thời gian: {self.recorded_time} giây (camera {self.current_cam_index})"
        self.info_label.config(text=status)
        self.show_behavior_report()
        self.save_behavior_report()

    def update_count_display(self):
        """Cập nhật label bên UI hiển thị kết quả đếm hiện tại."""
        if not self.counting:
            return
        if self.current_student:
            lines = [f"Học sinh hiện tại: {self.current_student}", "", "Số lần xuất hiện của các hành vi:"]
            beh = self.student_behavior_counts.get(self.current_student, {})
            if beh:
                for b, c in sorted(beh.items()):
                    lines.append(f"- {b}: {c} lần")
            else:
                lines.append("Chưa phát hiện hành vi nào.")
            cur = self.current_behaviors.get(self.current_student)
            if cur:
                lines.append("")
                lines.append(f"Hành vi hiện tại: {cur}")
        else:
            lines = ["Số lần xuất hiện của các hành vi:"]
            if self.behavior_counts:
                for b, c in sorted(self.behavior_counts.items()):
                    lines.append(f"- {b}: {c} lần")
            else:
                lines.append("Chưa phát hiện hành vi nào.")
            if self.student_detection_enabled.get():
                lines.append("")
                lines.append("(Đang đợi nhận diện học sinh...)")
        self.count_label.config(text="\n".join(lines))

    def save_behavior_report(self):
        """Lưu báo cáo text vào thư mục reports."""
        os.makedirs("reports", exist_ok=True)
        fname = time.strftime("reports/behavior_report_%Y%m%d_%H%M%S.txt")
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(f"BÁO CÁO HÀNH VI - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Thời gian đếm: {self.recorded_time} giây\n")
                f.write(f"Camera: {self.current_cam_index}\n\n")
                if self.student_detection_enabled.get() and self.student_behavior_counts:
                    f.write("BÁO CÁO THEO HỌC SINH:\n=====================\n\n")
                    for student, behaviors in sorted(self.student_behavior_counts.items()):
                        f.write(f"Học sinh: {student}\n")
                        if behaviors:
                            for b, c in sorted(behaviors.items()):
                                f.write(f"  - {b}: {c} lần\n")
                        else:
                            f.write("  Không phát hiện hành vi nào.\n")
                        f.write("\n")
                    f.write("\nTỔNG HỢP TẤT CẢ HÀNH VI:\n=====================\n\n")
                f.write("Số lần xuất hiện của các hành vi:\n")
                if self.behavior_counts:
                    for b, c in sorted(self.behavior_counts.items()):
                        f.write(f"- {b}: {c} lần\n")
                else:
                    f.write("Không phát hiện hành vi nào trong thời gian đếm.\n")
            messagebox.showinfo("Thông báo", f"Đã lưu báo cáo vào file: {fname}")
        except Exception as e:
            LOG.exception("Lỗi save_behavior_report: %s", e)
            messagebox.showerror("Lỗi", f"Không lưu được báo cáo: {e}")

    def show_behavior_report(self):
        """Hiển thị báo cáo chi tiết trong một cửa sổ mới."""
        report_window = tk.Toplevel(self.root)
        report_window.title("📊 Báo cáo hành vi")
        report_window.geometry("700x700")
        report_window.configure(bg="#f2e9e4")

        text_frame = tk.Frame(report_window, bg="#f2e9e4")
        text_frame.pack(fill="both", expand=True, padx=20, pady=20)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        text_widget = tk.Text(text_frame, wrap="word", font=("Arial", 12), bg="#ffffff")
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)

        report = []
        report.append("BÁO CÁO HÀNH VI\n")
        if hasattr(self, "recorded_time") and self.recorded_time:
            report.append(f"Thời gian đếm: {self.recorded_time} giây\n")

        if self.student_detection_enabled.get() and self.student_behavior_counts:
            report.append("\nBÁO CÁO THEO HỌC SINH:\n=====================\n")
            for student, behaviors in sorted(self.student_behavior_counts.items()):
                report.append(f"Học sinh: {student}\n")
                if behaviors:
                    for b, c in sorted(behaviors.items()):
                        report.append(f"  - {b}: {c} lần\n")
                else:
                    report.append("  Không phát hiện hành vi nào.\n")
                report.append("\n")
            report.append("\nTỔNG HỢP TẤT CẢ HÀNH VI:\n=====================\n")

        report.append("\nSố lần xuất hiện của các hành vi:\n")
        if self.behavior_counts:
            for b, c in sorted(self.behavior_counts.items()):
                report.append(f"- {b}: {c} lần\n")
        else:
            report.append("Không phát hiện hành vi nào trong thời gian đếm.")

        text_widget.insert("1.0", "".join(report))
        text_widget.config(state="disabled")

        # Save & Export buttons
        save_btn = tk.Button(report_window, text="Lưu báo cáo", command=self.save_behavior_report,
                             bg="#ffffff", activebackground="#fde2e4", relief="raised", bd=2, font=("Arial", 12, "bold"), cursor="hand2")
        save_btn.pack(pady=6)

        # Excel export if pandas available
        try:
            import pandas as pd  # noqa: F401
            excel_btn = tk.Button(report_window, text="Xuất Excel", command=self.export_excel_report,
                                  bg="#ffffff", activebackground="#d0f0c0", relief="raised", bd=2, font=("Arial", 12, "bold"), cursor="hand2")
            excel_btn.pack(pady=6)
        except Exception:
            pass

    def export_excel_report(self):
        """Xuất báo cáo sang Excel nếu pandas có sẵn."""
        try:
            import pandas as pd
            os.makedirs("reports", exist_ok=True)
            excel_file = time.strftime("reports/behavior_report_%Y%m%d_%H%M%S.xlsx")

            summary_data = [{"Hành vi": b, "Số lần xuất hiện": c} for b, c in sorted(self.behavior_counts.items())]
            summary_df = pd.DataFrame(summary_data)

            student_data = []
            for student, behaviors in sorted(self.student_behavior_counts.items()):
                for b, c in sorted(behaviors.items()):
                    student_data.append({"Học sinh": student, "Hành vi": b, "Số lần xuất hiện": c})
            student_df = pd.DataFrame(student_data)

            with pd.ExcelWriter(excel_file) as writer:
                summary_df.to_excel(writer, sheet_name="Tổng hợp", index=False)
                if not student_df.empty:
                    student_df.to_excel(writer, sheet_name="Theo học sinh", index=False)

            messagebox.showinfo("Thông báo", f"Đã xuất báo cáo Excel: {excel_file}")
        except ImportError:
            messagebox.showerror("Lỗi", "Cần cài đặt pandas để xuất Excel:\n pip install pandas openpyxl")
        except Exception as e:
            LOG.exception("Lỗi export_excel_report: %s", e)
            messagebox.showerror("Lỗi", f"Không thể xuất Excel: {e}")

    def update_camera(self):
        """Hàm được gọi định kỳ để đọc frame từ camera và hiển thị."""
        try:
            if self.cap and getattr(self.cap, "isOpened", lambda: False)() and self.main_frame.winfo_ismapped():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    h0, w0 = frame.shape[:2]
                    vw = max(320, self.video_label.winfo_width())
                    vh = max(240, self.video_label.winfo_height())
                    scale = min(vw / w0, vh / h0) if w0 and h0 else 1.0
                    new_w = max(320, int(w0 * scale))
                    new_h = max(240, int(h0 * scale))

                    resized_bgr = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # Student detection (throttled)
                    now = time.time()
                    if self.student_detection_enabled.get() and self.student_model is not None and (now - self.last_student_time > self.student_cooldown):
                        try:
                            student = self.detect_student(resized_bgr)
                            if student:
                                # ensure entry exists
                                if student not in self.student_behavior_counts:
                                    self.student_behavior_counts[student] = Counter()
                        except Exception:
                            LOG.exception("Lỗi khi detect_student trong loop")

                    # Object detection & counting
                    try:
                        annotated = self.detect_objects_and_count(resized_bgr)
                    except Exception:
                        LOG.exception("Lỗi detect_objects_and_count")
                        annotated = resized_bgr

                    # Convert BGR->RGB for Tk display
                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
                    self.frame_size = (new_w, new_h)

                    # update status label
                    status = f"🟢 Camera {self.current_cam_index} đang bật"
                    if self.counting:
                        status += " (Đang đếm hành vi...)"
                    else:
                        parts = []
                        if self.student_detection_enabled.get():
                            parts.append("Nhận diện học sinh: BẬT")
                        if self.detection_enabled.get():
                            parts.append("Nhận diện hành vi: BẬT")
                        status += f" ({', '.join(parts)})" if parts else " – sẵn sàng đếm hành vi."
                    self.info_label.config(text=status)
                else:
                    self.video_label.configure(image="")
        except Exception:
            LOG.exception("Lỗi update_camera")
        finally:
            # call again after ~30 ms (~33 fps)
            self.root.after(30, self.update_camera)

    def on_close(self):
        if self.counting:
            if not messagebox.askyesno("Đang đếm", "Đang đếm hành vi. Muốn dừng và thoát?"):
                return
            self.stop_counting()
        try:
            if self.cap and getattr(self.cap, "isOpened", lambda: False)():
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()

    def __del__(self):
        try:
            if self.cap and getattr(self.cap, "isOpened", lambda: False)():
                self.cap.release()
        except Exception:
            pass
document.addEventListener("DOMContentLoaded", () => {
  const menuToggle = document.querySelector(".menu-toggle");
  const navLinks = document.querySelector(".nav-links");

  if (menuToggle && navLinks) {
    menuToggle.addEventListener("click", () => {
      const open = navLinks.classList.toggle("open");
      menuToggle.setAttribute("aria-expanded", String(open));
    });
  }

  const profileMenu = document.querySelector(".profile-menu");
  const profileTrigger = document.querySelector(".profile-trigger");

  if (profileMenu && profileTrigger) {
    profileTrigger.addEventListener("click", (event) => {
      event.stopPropagation();
      const open = profileMenu.classList.toggle("open");
      profileTrigger.setAttribute("aria-expanded", String(open));
    });

    document.addEventListener("click", (event) => {
      if (!profileMenu.contains(event.target)) {
        profileMenu.classList.remove("open");
        profileTrigger.setAttribute("aria-expanded", "false");
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        profileMenu.classList.remove("open");
        profileTrigger.setAttribute("aria-expanded", "false");
        profileTrigger.focus();
      }
    });
  }

  document.querySelectorAll(".password-toggle").forEach((button) => {
    button.addEventListener("click", () => {
      const input = button.parentElement.querySelector("input");
      const showing = input.type === "text";
      input.type = showing ? "password" : "text";
      button.textContent = showing ? "Show" : "Hide";
    });
  });

  const uploadForm = document.getElementById("upload-form");
  const input = document.getElementById("video-input");
  const dropZone = document.getElementById("drop-zone");
  const browseButton = document.getElementById("browse-button");
  const previewPanel = document.getElementById("file-preview");
  const previewVideo = document.getElementById("video-preview");
  const fileName = document.getElementById("file-name");
  const fileSize = document.getElementById("file-size");
  const removeButton = document.getElementById("remove-file");
  const overlay = document.getElementById("analysis-overlay");
  let previewUrl = null;

  if (!uploadForm || !input || !dropZone) {
    return;
  }

  const openFilePicker = () => input.click();

  const showFile = (file) => {
    if (!file || !file.type.startsWith("video/")) {
      return;
    }

    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);
    previewVideo.src = previewUrl;
    fileName.textContent = file.name;
    fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    dropZone.hidden = true;
    previewPanel.hidden = false;
  };

  browseButton.addEventListener("click", (event) => {
    event.stopPropagation();
    openFilePicker();
  });

  dropZone.addEventListener("click", openFilePicker);
  dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openFilePicker();
    }
  });

  input.addEventListener("change", () => showFile(input.files[0]));

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("dragging");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("dragging");
    });
  });

  dropZone.addEventListener("drop", (event) => {
    showFile(event.dataTransfer.files[0]);
  });

  removeButton.addEventListener("click", () => {
    input.value = "";
    previewVideo.removeAttribute("src");
    previewVideo.load();
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    previewPanel.hidden = true;
    dropZone.hidden = false;
  });

  uploadForm.addEventListener("submit", () => {
    overlay.hidden = false;
  });

  const liveStreamBtn = document.getElementById("live-stream-btn");
  const recordingOverlay = document.getElementById("recording-overlay");
  const recordingCloseBtn = document.getElementById("recording-close-btn");
  const recordingPreviewVideo = document.getElementById("recording-preview-video");
  const recordingStatusText = document.getElementById("recording-status-text");
  const recordingCountdown = document.getElementById("recording-countdown-timer");
  const recordingStartBtn = document.getElementById("recording-start-btn");
  const recordingCancelBtn = document.getElementById("recording-cancel-btn");
  const analysisOverlay = document.getElementById("analysis-overlay");

  if (!liveStreamBtn || !recordingOverlay) return;

  let mediaRecorder = null;
  let mediaStream = null;
  let recordedChunks = [];
  let countdownInterval = null;
  let recordingCancelled = false;

  liveStreamBtn.addEventListener("click", async () => {
    recordingOverlay.hidden = false;
    recordingStatusText.textContent = "Requesting camera access...";
    recordingStartBtn.disabled = true;

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      recordingPreviewVideo.srcObject = mediaStream;
      recordingStatusText.textContent = "Camera ready";
      recordingStartBtn.disabled = false;
    } catch (err) {
      recordingStatusText.textContent = "Camera access denied or unavailable";
    }
  });

  const closeRecording = () => {
    recordingCancelled = true;
    if (countdownInterval) clearInterval(countdownInterval);
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
    if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
    recordingOverlay.hidden = true;
    recordingPreviewVideo.srcObject = null;
    recordingCountdown.hidden = true;
    recordingStartBtn.disabled = true;
    recordedChunks = [];
  };

  recordingCloseBtn.addEventListener("click", closeRecording);
  recordingCancelBtn.addEventListener("click", closeRecording);

  recordingStartBtn.addEventListener("click", () => {
    recordedChunks = [];
    recordingCancelled = false;
    recordingStartBtn.disabled = true;
    recordingCloseBtn.disabled = true;
    recordingCancelBtn.disabled = true;
    recordingCountdown.hidden = false;
    recordingCountdown.textContent = "10";

    mediaRecorder = new MediaRecorder(mediaStream, {
      mimeType: MediaRecorder.isTypeSupported("video/webm;codecs=vp9,opus")
        ? "video/webm;codecs=vp9,opus"
        : "video/webm",
    });

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };

    let remaining = 10;
    countdownInterval = setInterval(() => {
      remaining--;
      recordingCountdown.textContent = String(remaining);
      if (remaining <= 0) {
        clearInterval(countdownInterval);
        countdownInterval = null;
      }
    }, 1000);

    mediaRecorder.onstop = async () => {
      if (countdownInterval) {
        clearInterval(countdownInterval);
        countdownInterval = null;
      }

      if (recordingCancelled) {
        recordingCancelled = false;
        return;
      }

      const blob = new Blob(recordedChunks, { type: "video/webm" });
      const formData = new FormData();
      const filename = `live_recording_${Date.now()}.webm`;
      formData.append("video", blob, filename);

      recordingOverlay.hidden = true;
      recordingPreviewVideo.srcObject = null;
      recordingStartBtn.disabled = true;
      recordingCloseBtn.disabled = false;
      recordingCancelBtn.disabled = false;

      if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());

      analysisOverlay.hidden = false;

      try {
        const response = await fetch("/upload", {
          method: "POST",
          body: formData,
        });
        if (response.redirected) {
          window.location.href = response.url;
        } else {
          document.open();
          document.write(await response.text());
          document.close();
        }
      } catch (err) {
        analysisOverlay.hidden = true;
        alert("Upload failed. Please try again.");
      }
    };

    mediaRecorder.start();
    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
    }, 10000);
  });
});

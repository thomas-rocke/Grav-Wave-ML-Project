conda activate ML-GPU

try { & "BG(3,3).ps1" } catch { "Program failed! "}
try { & "D(3).ps1" } catch { "Program failed! "}
try { & "BG(5,5).ps1" } catch { "Program failed! "}
try { & "D(5).ps1" } catch { "Program failed! "}
try { & "Multiprocessing.ps1" } catch { "Program failed! "}
try { & "Optimise_BG(3,3).ps1" } catch { "Program failed! "}
try { & "Optimise_D(3).ps1" } catch { "Program failed! "}
try { & "Optimise_BG(5,5).ps1" } catch { "Program failed! "}
try { & "Optimise_D(5).ps1" } catch { "Program failed! "}
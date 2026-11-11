import tkinter as tk

root = tk.Tk()
root.title("Test Tkinter")
root.geometry("400x300")
root.configure(bg="#1e1e1e")

etiqueta = tk.Label(root, text="¡Tkinter funciona!", fg="white", bg="#1e1e1e", font=("Arial", 16))
etiqueta.pack(expand=True)

root.after(2000, root.destroy) # Cierra la ventana tras 2 segundos
root.mainloop()
print("Tkinter verificado correctamente.")
def main():
    import smtplib

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("davidjamesminidemo@gmail.com", "paypaldemo")

    msg = "Security alert!!!"
    server.sendmail("davidjamesminidemo@gmail.com", "davidjamesminidemo@gmail.com", msg)
    server.quit()
    print('Security alert has been sent out.')

if __name__ == "__main__":
    main()

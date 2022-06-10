from timeit import default_timer as timer



class _loading_bar:
    
    def __init__(self, count, headline):
        self.count = count
        self.current = 0
        self.skip = 0
        self.start = timer()
        print("\n{0} in progress...".format(headline))
        
    def _update(self, loss = None):
        self.current += 1
        if self.current == self.count:
            print("\r{0}".format(f"Finished Successfully after {round(timer()-self.start,1)} seconds!                               \n"))
        elif self.skip > self.count/133:
            remaining = (self.count/self.current)*(timer()-self.start)-(timer()-self.start)
            if remaining > 60:
                remaining = f"{round(remaining/60,1)} minutes remaining"
            else:
                remaining = f"{round(remaining,1)} seconds remaining"
            if not loss == None:
                print("\r{0}".format("|{0}{1}|   {2}% finished // {3} (Loss: {4})".format("█"*int(round(self.current/self.count,1)*20),"-"*(20-int(round(self.current/self.count,1)*20)),round((self.current/self.count)*100,2),remaining, round(loss))), end = "", flush=True)
            print("\r{0}".format("|{0}{1}|   {2}% finished // {3}".format("█"*int(round(self.current/self.count,1)*20),"-"*(20-int(round(self.current/self.count,1)*20)),round((self.current/self.count)*100,2),remaining)), end = "", flush=True)
            self.skip = 0
        else:
            self.skip += 1

    def _kill(self):
        print("\r{0}".format(f"Finished Successfully after {round(timer()-self.start,1)} seconds!                               \n"))
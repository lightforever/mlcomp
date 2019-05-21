import { Injectable } from '@angular/core';
import {BaseService} from "./base.service";
import {AppSettings} from "./app-settings";
import {catchError, tap} from "rxjs/operators";
import {PaginatorRes, Status} from "./models";

@Injectable({
  providedIn: 'root'
})
export class TaskService extends BaseService{
   protected collection_part: string = 'tasks';
   protected single_part: string = 'task';


    stop(id: number) {
        let message = `${this.constructor.name}.stop`;
        return this.http.post<Status>(AppSettings.API_ENDPOINT + this.single_part+'/stop', {'id': id}).pipe(
            catchError(this.handleError<Status>(message, new Status()))
        );
    }
}

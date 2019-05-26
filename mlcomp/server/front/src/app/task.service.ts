import { Injectable } from '@angular/core';
import {BaseService} from "./base.service";
import {AppSettings} from "./app-settings";
import {catchError, tap} from "rxjs/operators";
import {PaginatorRes, Status, StepNode, ToogleReportResult} from "./models";

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

    toogle_report(id: any, report: number, report_full: boolean) {
        let message = `${this.constructor.name}.toogle_report`;
        return this.http.post<ToogleReportResult>(AppSettings.API_ENDPOINT + this.single_part + '/toogle_report', {
            'id': id,
            'report': report,
            'remove': report_full
        }).pipe(
            catchError(this.handleError<ToogleReportResult>(message, new ToogleReportResult()))
        );
    }

    steps(id: number) {
        let message = `${this.constructor.name}.steps`;
        return this.http.post<StepNode[]>(AppSettings.API_ENDPOINT + this.single_part+'/steps', id).pipe(
            catchError(this.handleError<StepNode[]>(message, []))
        );
    }
}

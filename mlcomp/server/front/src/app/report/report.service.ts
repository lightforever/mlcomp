import {EventEmitter, Injectable} from '@angular/core';
import {BaseService} from "../base.service";
import {BaseResult, ReportAddData} from "../models";
import {AppSettings} from "../app-settings";
import {catchError} from "rxjs/operators";

@Injectable({
    providedIn: 'root'
})
export class ReportService extends BaseService {
    protected collection_part: string = 'reports';
    protected single_part: string = 'report';

    data_updated: EventEmitter<any> = new EventEmitter();

    add_start() {
        let message = `${this.constructor.name}.add_start`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/add_start';
        return this.http.post<BaseResult>(url, {}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    add_end(data: ReportAddData) {
        let message = `${this.constructor.name}.add_end`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/add_end';
        return this.http.post<BaseResult>(url, data).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }
}

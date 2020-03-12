import {BaseService} from "../../base.service";
import {Injectable} from "@angular/core";
import {BaseResult, Memory} from "../../models";
import {AppSettings} from "../../app-settings";
import {catchError} from "rxjs/operators";

@Injectable({
    providedIn: 'root'
})
export class MemoryService extends BaseService {
    protected collection_part: string = 'memories';
    protected single_part: string = 'memory';

    add(data: Memory) {
        let message = `${this.constructor.name}.add`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/add';
        return this.http.post<BaseResult>(url, data).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    edit(data: Memory) {
        let message = `${this.constructor.name}.edit`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/edit';
        return this.http.post<BaseResult>(url, data).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    remove(id: number) {
        let message = `${this.constructor.name}.remove`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/remove';
        return this.http.post<BaseResult>(url, {'id': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

}